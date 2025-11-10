from digirl.environment import batch_interact_environment
from digirl.data import ReplayBuffer
import numpy as np
from tqdm import tqdm
from offline_rl import OfflineRLTrainer
from misc import colorful_print
import wandb
import os
import torch
import time
import copy
from env_utils import add_mc_return

def label_trajectories(trajectories, agent):
    """
    Label trajectories using the agent's trajectory critic to compute baseline values.
    
    Args:
        trajectories: List of trajectory data
        agent: Agent with trajectory_critic method
        
    Returns:
        torch.Tensor: Clamped baseline values for each trajectory
    """
    print("Labeling Trajectories")
    baselines = []
    
    # Process trajectories in batches of 16 for efficiency
    for i in range(0, len(trajectories), 16):
        observations = [t[0]["observation"] for t in trajectories[i:i+16]]
        with torch.no_grad():
            # Get trajectory critic values and apply softmax
            v = agent.trajectory_critic(observations)
            v = torch.nn.Softmax(dim = -1)(v)[:,1]  # Take second column after softmax
            baselines.append(v.flatten())
    
    baselines = torch.cat(baselines, dim = -1)
    print("Done Labeling Trajectories")
    # Clamp values to avoid numerical instability
    return torch.clamp(baselines.cpu(), 1e-4, 1-1e-4)

def framestack(all_trajectories):
    """
    Apply frame stacking to trajectories by concatenating current and previous image features.
    
    Args:
        all_trajectories: List of trajectories containing image features
        
    Returns:
        List of trajectories with frame-stacked image features
    """
    new_trajectories = copy.deepcopy(all_trajectories)
    
    for trajectory, new_trajectory in zip(all_trajectories, new_trajectories):
        for i,(t, nt) in enumerate(zip(trajectory, new_trajectory)):
            if i == 0:
                # For first frame, duplicate current image features
                nt["image_features"] = np.concatenate([t["image_features"], t["image_features"]], axis = -1)
            else:
                # Stack previous and current image features
                nt["image_features"] = np.concatenate([trajectory[i-1]["image_features"], t["image_features"]], axis = -1)
            # Stack current and next image features
            nt["next_image_features"] = np.concatenate([t["image_features"], t["next_image_features"]], axis = -1)
    
    return new_trajectories

def filterbc_buffer(all_trajectories, batch_size, capacity, agent):
    """
    Create a filtered replay buffer using only top 10% trajectories by reward (Behavioral Cloning filter).
    
    Args:
        all_trajectories: List of all trajectories
        batch_size: Batch size for replay buffer
        capacity: Maximum capacity of replay buffer
        agent: Agent (unused in current implementation)
        
    Returns:
        ReplayBuffer: Filtered buffer containing only high-reward trajectories
    """
    # Extract trajectory rewards, handling empty trajectories
    trajectory_rewards = np.array([t[0]["trajectory_reward"] if len(t) > 0 else 0 for t in all_trajectories]).flatten()
    
    # Calculate 90th percentile cutoff (top 10%)
    cutoff = np.quantile(trajectory_rewards, 1 - 0.1)
    
    # Print top 10 trajectories for debugging
    top10 = np.argsort(trajectory_rewards)[-10:]
    print("Top 10 Trajectories: ")
    for i in top10:
        print(all_trajectories[i][0]["observation"])
        print(trajectory_rewards[i])
    print("Cutoff: ", cutoff)
    
    # Filter trajectories above cutoff
    filtered_trajectories = []
    for t, b in zip(all_trajectories, trajectory_rewards):
        if b >= cutoff:
            filtered_trajectories.append(t)
    
    # Flatten trajectory data and create replay buffer
    data = sum(filtered_trajectories, [])
    filtered_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    for d in data:
        filtered_buffer.insert(**d)
    
    return filtered_buffer

def offpolicy_train_loop(env,
                agent,
                tokenizer,
                accelerator,
                warmup_iter: int = 20,
                rollout_size: int = 50,
                batch_size: int = 2,
                capacity: int = 500000,
                train_iterations: int = 10,
                epochs: int = 3, 
                grad_accum_steps: int = 1,
                critic_lr: float = 1e-3,
                lm_lr: float = 1e-5,
                gamma: float = 0.9,
                tau: float = 0.1,
                use_wandb: bool = False,
                actor_epochs: int = 3,
                train_mode: str = None,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                save_freq: int = 25,
                train_algorithm: str = "offlinerl",
                decode_f: callable = lambda x: x,
                offline_data_path: str = None,
                offline_actor_iterations: int = 20,
                offline_critic_iterations: int = 20,
                offline_trajectory_critic_iterations: int = 20,
                trajectory_critic_epochs: int = 5,
                parallel: str = 'single',
                worker_temp_path=None, 
                worker_run_path=None,
                worker_ips=[], 
                worker_username=None,
                **kwargs):
    """
    Main training loop for offline reinforcement learning with GUI agents.
    Supports both offline pretraining and online fine-tuning phases.
    
    Args:
        env: Environment for interaction
        agent: RL agent to train
        tokenizer: Text tokenizer
        accelerator: Distributed training accelerator
        warmup_iter: Number of warmup iterations before actor updates
        rollout_size: Number of trajectories per rollout
        batch_size: Training batch size
        capacity: Replay buffer capacity
        train_iterations: Total training iterations
        ... (other parameters documented inline)
    """

    # Initialize trainer with specified hyperparameters
    trainer = OfflineRLTrainer(agent=agent,
                            tokenizer=tokenizer,
                            accelerator=accelerator,
                            lm_lr=lm_lr,
                            epochs=actor_epochs,
                            grad_accum_steps=grad_accum_steps,
                            max_grad_norm=max_grad_norm)
    
    # Initialize replay buffer and trajectory storage
    replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    all_trajectories = []
    
    # Prepare model and optimizers for distributed training
    agent.prepare()
    trainer.prepare()

    loaded_trajs = False
    
    # ==================== DATA LOADING PHASE ====================
    
    # Load offline data if provided
    if offline_data_path is not None:
        print("Loading offline trajectories...")
        all_trajectories = torch.load(offline_data_path)
        all_trajectories = framestack(all_trajectories)  # Apply frame stacking
        print(f"The number of offline trajectories is {len(all_trajectories)}")
        
        # Add Monte Carlo returns to trajectories
        all_trajectories = [add_mc_return(t, gamma=gamma) for t in all_trajectories]
        
        # Split into train/validation sets (80/20)
        train_trajectories = all_trajectories[:int(len(all_trajectories)*0.8)]
        val_trajectories = all_trajectories[int(len(all_trajectories)*0.8):]
        loaded_trajs = 'scratch'
        
    # Resume from checkpoint if available (only for online/off2on training)
    if os.path.exists(os.path.join(save_path, 'trainer.pt')):
        assert train_mode != "offline", "Only online/off2on training can be resumed"
        print("Resuming from checkpoint...")
        
        # Load all saved components
        trainer.load(os.path.join(save_path, 'trainer.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
        train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'))
        val_trajectories = torch.load(os.path.join(save_path, 'val_trajectories.pt'))
        
        print(f"The number of online trajectories is {len(all_trajectories)}")
        if use_wandb and accelerator.is_main_process:
            print("Loading from checkpoint")
        loaded_trajs = 'resume'
            
    # Initialize empty trajectory lists if no data loaded
    if not loaded_trajs:
        train_trajectories = []
        val_trajectories = []
        all_trajectories = []

    # Reinitialize buffers and populate with loaded data
    replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)

    # Flatten trajectories and populate buffers
    data = sum(train_trajectories, [])
    val_data = sum(val_trajectories, [])
    for d in data:
        replay_buffer.insert(**d)
    for d in val_data:
        validation_buffer.insert(**d)
    
    # ==================== OFFLINE TRAINING PHASE ====================
    
    # Only train offline if no main trainer checkpoint exists
    if not os.path.exists(os.path.join(save_path, 'trainer.pt')):
        # Try to load existing offline trainer
        if os.path.exists(os.path.join(save_path, 'trainer_offline.pt')):
            trainer.load(os.path.join(save_path, 'trainer_offline.pt'))
            print("Loading from offline trainer")
        else:
            # Perform offline training if data is available
            if offline_data_path is not None:
                print(">>>Training Offline")
                info = {}
                
                # Create filtered buffers using top 10% trajectories
                # Note: Offline training uses filterbc_buffer, not trajectory-level critic filter
                filtered_buffer = filterbc_buffer(train_trajectories, batch_size, capacity, agent)
                filtered_validation_buffer = filterbc_buffer(val_trajectories, batch_size, capacity, agent)
                
                # Behavioral Cloning training phase
                print("Starting filtered BC training...")
                for i in tqdm(range(offline_actor_iterations), disable=not accelerator.is_main_process):
                    info.update(trainer.update(filtered_buffer))
                    if use_wandb and accelerator.is_main_process:
                        wandb.log(info)
                
                # Save offline trainer
                if accelerator.is_main_process:
                    trainer.save(os.path.join(save_path, 'trainer_offline.pt'))
                    print("Offline training completed and saved")

    # ==================== ONLINE TRAINING PHASE ====================
    
    if accelerator.is_main_process:
        print(">>>start iterations")
    
    # Calculate starting iteration for resuming
    if loaded_trajs == "resume":
        resume_iter = len(all_trajectories) // rollout_size
    else:
        resume_iter = 0
    
    # Initialize progress bar
    progress_bar = tqdm(total=train_iterations, initial=resume_iter)
    
    # Main training loop
    for i in range(resume_iter, train_iterations):
        # Ensure we're not in offline-only mode for iterative training
        assert train_mode != "offline", "Only online/off2on need to iteractively train; offline should directly go to eval loop after training"
        
        # =============== TRAJECTORY COLLECTION ===============
        
        if parallel == 'single':
            # Single-process trajectory collection
            trajectories = batch_interact_environment(agent=agent,
                                                env=env,
                                                num_trajectories=rollout_size,
                                                accelerator=accelerator,
                                                use_tqdm=False,
                                                decode_f=decode_f,
                                                gamma=gamma,
                                                iter=i)
        elif parallel == 'host':
            # Distributed trajectory collection using remote workers
            if i == 0:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            trajectories = remote_collect_trajectories(save_path=save_path, 
                                                       worker_temp_path=worker_temp_path, 
                                                       worker_run_path=worker_run_path,
                                                       worker_ips=worker_ips, 
                                                       worker_username=worker_username,
                                                       trainer=trainer)
        
        # Apply frame stacking to collected trajectories
        trajectories = framestack(trajectories)
        
        # =============== DATA PROCESSING & STORAGE ===============
        
        if accelerator.is_main_process:
            # Log trajectory statistics
            trajectory_rewards = [d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]
            info = {"iteration": i,
                    "rollout.mean": np.mean(trajectory_rewards),
                    "rollout.max": np.max(trajectory_rewards),
                    "rollout.min": np.min(trajectory_rewards),
                    "walltime": time.time()}
            
            # Update trajectory collections
            all_trajectories += trajectories
            colorful_print(f">>> length of all_trajectories: {len(trajectories)}", fg='green')
            
            # Split new trajectories into train/val (80/20)
            new_train_trajectories = trajectories[:int(len(trajectories)*0.8)]
            new_val_trajectories = trajectories[int(len(trajectories)*0.8):]
            train_trajectories += new_train_trajectories
            val_trajectories += new_val_trajectories
            
            # Flatten and add to replay buffers
            data = sum(new_train_trajectories, [])
            val_data = sum(new_val_trajectories, [])
            for d in data:
                replay_buffer.insert(**d)
            for d in val_data:
                validation_buffer.insert(**d)
        
            # Log step-level reward statistics
            if data:  # Only if we have data
                info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),
                        "rollout.reward.max": np.max([d["reward"] for d in data]),
                        "rollout.reward.min": np.min([d["reward"] for d in data])})
            
            # Save all data to disk
            print(">>> Saving Replay Buffer")
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
            torch.save(train_trajectories, os.path.join(save_path, 'train_trajectories.pt'))
            torch.save(val_trajectories, os.path.join(save_path, 'val_trajectories.pt'))
            print(">>> Saved Replay Buffer")
            time.sleep(15)  # Brief pause for I/O completion
        else:
            info = {}
        
        # Synchronize all processes
        accelerator.wait_for_everyone()
        
        # =============== LOAD DATA FOR TRAINING ===============
        
        # All processes load the saved data for consistent training
        train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'))
        val_trajectories = torch.load(os.path.join(save_path, 'val_trajectories.pt'))
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))

        # =============== TRAINING STEP ===============
        
        # Validate training algorithm
        assert train_algorithm in ['digirl', 'filteredbc'], "Only digirl and filteredbc are supported"
        
        # Create filtered buffers for training
        filtered_buffer = filterbc_buffer(train_trajectories, batch_size, capacity, agent)
        filtered_validation_buffer = filterbc_buffer(val_trajectories, batch_size, capacity, agent)
           
        print("Training")
        if 'filtered' in train_algorithm:
            # Filtered BC algorithm: only use top trajectories
            info.update(trainer.update(filtered_buffer, no_update_actor=(i < warmup_iter)))
            del filtered_buffer  # Free memory
        else:
            # DigiRL algorithm: use trajectory critic + full buffer training
            info.update(trainer.update_trajectory_critic(train_trajectories, val_trajectories))
            info.update(trainer.update(replay_buffer, validation_buffer, filtered_buffer, filtered_validation_buffer, no_update_actor=(i < warmup_iter)))
        
        # =============== LOGGING & CHECKPOINTING ===============
        
        # Log training metrics
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
            
        # Periodic saving
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            
        # Update progress bar
        if accelerator.is_main_process:
            progress_bar.update(1)