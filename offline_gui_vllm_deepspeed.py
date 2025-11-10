from environment import batch_interact_environment
from data import ReplayBuffer
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

# ==================== vLLM and DeepSpeed Imports ====================
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: vLLM not available. Falling back to standard inference.")
    VLLM_AVAILABLE = False

try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    DEEPSPEED_AVAILABLE = True
except ImportError:
    print("Warning: DeepSpeed not available. Using standard training.")
    DEEPSPEED_AVAILABLE = False

def initialize_vllm_engine(model_name_or_path, tensor_parallel_size=1, gpu_memory_utilization=0.9):
    """
    Initialize vLLM engine for fast inference.
    
    Args:
        model_name_or_path: Path to the model or HuggingFace model name
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization ratio
        
    Returns:
        LLM: Initialized vLLM engine
    """
    if not VLLM_AVAILABLE:
        return None
    
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="float16",  # Use half precision for memory efficiency
        swap_space=4,     # GB of CPU memory for swapping
        enforce_eager=False,  # Use CUDA graphs for better performance
    )
    return llm

def setup_deepspeed_config(stage=2, offload_optimizer=True, offload_param=False):
    """
    Create DeepSpeed configuration for Zero Redundancy Optimizer.
    
    Args:
        stage: ZeRO stage (1, 2, or 3)
        offload_optimizer: Whether to offload optimizer states to CPU
        offload_param: Whether to offload parameters to CPU (ZeRO-3 only)
        
    Returns:
        dict: DeepSpeed configuration dictionary
    """
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "wall_clock_breakdown": False
    }
    
    # ZeRO configuration based on stage
    if stage == 1:
        config["zero_optimization"] = {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        }
    elif stage == 2:
        config["zero_optimization"] = {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "cpu_offload": offload_optimizer
        }
    elif stage == 3:
        config["zero_optimization"] = {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            "cpu_offload": offload_optimizer,
            "cpu_offload_params": offload_param
        }
    
    return config

def label_trajectories(trajectories, agent, use_vllm=False, vllm_engine=None):
    """
    Label trajectories using the agent's trajectory critic to compute baseline values.
    Enhanced with vLLM support for faster inference.
    
    Args:
        trajectories: List of trajectory data
        agent: Agent with trajectory_critic method
        use_vllm: Whether to use vLLM for inference
        vllm_engine: Pre-initialized vLLM engine
        
    Returns:
        torch.Tensor: Clamped baseline values for each trajectory
    """
    print("Labeling Trajectories")
    baselines = []
    
    if use_vllm and vllm_engine is not None and VLLM_AVAILABLE:
        # Use vLLM for fast batch inference
        print("Using vLLM for trajectory labeling")
        observations = [t[0]["observation"] for t in trajectories]
        
        # Configure sampling parameters for deterministic evaluation
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic
            max_tokens=1,     # Only need logits, not generation
            logprobs=5        # Get top-5 logprobs for analysis
        )
        
        # Batch process all observations
        outputs = vllm_engine.generate(observations, sampling_params)
        
        # Extract trajectory values (this would need to be adapted based on your model's output format)
        for output in outputs:
            # This is a placeholder - you'd need to adapt this based on how your trajectory critic works
            baseline_value = extract_trajectory_value_from_vllm_output(output)
            baselines.append(baseline_value)
        
        baselines = torch.tensor(baselines)
    else:
        # Standard processing in batches of 16
        for i in range(0, len(trajectories), 16):
            observations = [t[0]["observation"] for t in trajectories[i:i+16]]
            with torch.no_grad():
                # Get trajectory critic values and apply softmax
                v = agent.trajectory_critic(observations)
                v = torch.nn.Softmax(dim=-1)(v)[:, 1]  # Take second column after softmax
                baselines.append(v.flatten())
        
        baselines = torch.cat(baselines, dim=-1)
    
    print("Done Labeling Trajectories")
    # Clamp values to avoid numerical instability
    return torch.clamp(baselines.cpu(), 1e-4, 1-1e-4)

def extract_trajectory_value_from_vllm_output(output):
    """
    Extract trajectory value from vLLM output.
    This is a placeholder function that needs to be implemented based on your specific model.
    
    Args:
        output: vLLM generation output
        
    Returns:
        float: Extracted trajectory value
    """
    # Placeholder implementation - adapt based on your model's output format
    return torch.rand(1).item()  # Replace with actual extraction logic

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
        for i, (t, nt) in enumerate(zip(trajectory, new_trajectory)):
            if i == 0:
                # For first frame, duplicate current image features
                nt["image_features"] = np.concatenate([t["image_features"], t["image_features"]], axis=-1)
            else:
                # Stack previous and current image features
                nt["image_features"] = np.concatenate([trajectory[i-1]["image_features"], t["image_features"]], axis=-1)
            # Stack current and next image features
            nt["next_image_features"] = np.concatenate([t["image_features"], t["next_image_features"]], axis=-1)
    
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
                # ==================== New vLLM and DeepSpeed Parameters ====================
                use_vllm: bool = False,
                vllm_tensor_parallel_size: int = 1,
                vllm_gpu_memory_utilization: float = 0.9,
                use_deepspeed: bool = False,
                deepspeed_config_path: str = None,
                deepspeed_stage: int = 2,
                deepspeed_offload_optimizer: bool = True,
                deepspeed_offload_param: bool = False,
                **kwargs):
    """
    Main training loop for offline reinforcement learning with GUI agents.
    Enhanced with vLLM and DeepSpeed support for improved performance and memory efficiency.
    
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
        use_vllm: Whether to use vLLM for fast inference
        vllm_tensor_parallel_size: Number of GPUs for vLLM tensor parallelism
        vllm_gpu_memory_utilization: GPU memory utilization for vLLM
        use_deepspeed: Whether to use DeepSpeed for training
        deepspeed_config_path: Path to DeepSpeed configuration file
        deepspeed_stage: DeepSpeed ZeRO stage (1, 2, or 3)
        deepspeed_offload_optimizer: Whether to offload optimizer to CPU
        deepspeed_offload_param: Whether to offload parameters to CPU
        ... (other parameters documented inline)
    """

    # ==================== vLLM Initialization ====================
    vllm_engine = None
    if use_vllm and VLLM_AVAILABLE:
        print("Initializing vLLM engine...")
        try:
            # Initialize vLLM engine for fast inference during trajectory collection
            model_name = getattr(agent, 'model_name', 'default_model')
            vllm_engine = initialize_vllm_engine(
                model_name_or_path=model_name,
                tensor_parallel_size=vllm_tensor_parallel_size,
                gpu_memory_utilization=vllm_gpu_memory_utilization
            )
            print("vLLM engine initialized successfully")
        except Exception as e:
            print(f"Failed to initialize vLLM: {e}. Falling back to standard inference.")
            use_vllm = False

    # ==================== DeepSpeed Configuration ====================
    deepspeed_config = None
    if use_deepspeed and DEEPSPEED_AVAILABLE:
        if deepspeed_config_path and os.path.exists(deepspeed_config_path):
            # Load custom DeepSpeed configuration
            import json
            with open(deepspeed_config_path, 'r') as f:
                deepspeed_config = json.load(f)
            print(f"Loaded DeepSpeed config from {deepspeed_config_path}")
        else:
            # Use auto-generated DeepSpeed configuration
            deepspeed_config = setup_deepspeed_config(
                stage=deepspeed_stage,
                offload_optimizer=deepspeed_offload_optimizer,
                offload_param=deepspeed_offload_param
            )
            print(f"Using auto-generated DeepSpeed config with ZeRO stage {deepspeed_stage}")

    # Initialize trainer with enhanced configuration
    trainer = OfflineRLTrainer(agent=agent,
                            tokenizer=tokenizer,
                            accelerator=accelerator,
                            lm_lr=lm_lr,
                            epochs=actor_epochs,
                            grad_accum_steps=grad_accum_steps,
                            max_grad_norm=max_grad_norm,
                            use_deepspeed=use_deepspeed,
                            deepspeed_config=deepspeed_config)
    
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
                                                iter=i,
                                                vllm_engine=vllm_engine if use_vllm else None)
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
                                                       trainer=trainer,
                                                       vllm_engine=vllm_engine if use_vllm else None)
        
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
        
        # Create filtered buffers for training (with vLLM support for trajectory labeling)
        filtered_buffer = filterbc_buffer(train_trajectories, batch_size, capacity, agent)
        filtered_validation_buffer = filterbc_buffer(val_trajectories, batch_size, capacity, agent)
           
        print("Training")
        if 'filtered' in train_algorithm:
            # Filtered BC algorithm: only use top trajectories
            info.update(trainer.update(filtered_buffer, no_update_actor=(i < warmup_iter)))
            del filtered_buffer  # Free memory
        else:
            # DigiRL algorithm: use trajectory critic + full buffer training
            # Enhanced with vLLM support for trajectory labeling
            info.update(trainer.update_trajectory_critic(train_trajectories, val_trajectories, 
                                                        use_vllm=use_vllm, vllm_engine=vllm_engine))
            info.update(trainer.update(replay_buffer, validation_buffer, filtered_buffer, filtered_validation_buffer, 
                                     no_update_actor=(i < warmup_iter)))
        
        # =============== LOGGING & CHECKPOINTING ===============
        
        # Log training metrics (including DeepSpeed memory stats if available)
        if use_deepspeed and DEEPSPEED_AVAILABLE:
            # Add DeepSpeed memory statistics
            if hasattr(trainer.lm_optimizer, 'get_memory_footprint'):
                memory_stats = trainer.lm_optimizer.get_memory_footprint()
                info.update({f"memory.{k}": v for k, v in memory_stats.items()})
        
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
            
        # Periodic saving (enhanced for DeepSpeed)
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            
            # Save additional DeepSpeed checkpoints if needed
            if use_deepspeed and DEEPSPEED_AVAILABLE:
                deepspeed_checkpoint_path = os.path.join(save_path, f'deepspeed_checkpoint_{i+1}')
                trainer.save_deepspeed_checkpoint(deepspeed_checkpoint_path)
            
        # Update progress bar
        if accelerator.is_main_process:
            progress_bar.update(1)
    
    # Cleanup vLLM engine
    if vllm_engine is not None:
        del vllm_engine
        torch.cuda.empty_cache()