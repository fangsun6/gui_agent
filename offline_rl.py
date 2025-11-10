import torch
import transformers
from tqdm import tqdm
import copy
import random
from torch.utils.data import DataLoader
from digirl.data import DummyDataset

def dict_mean(dict_list):
    """Calculate the mean of values across a list of dictionaries with the same keys."""
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

class OfflineRLTrainer():
    """
    Behavioral Cloning Trainer for offline reinforcement learning.
    Trains an agent to imitate expert behavior using supervised learning on pre-collected data.
    """
    
    def __init__(self, agent,
                 tokenizer,
                 accelerator,
                 lm_lr: float = 1e-5,
                 epochs: int = 3,
                 max_grad_norm: float = 0.01,
                 grad_accum_steps: int = 8):
        """
        Initialize the OfflineRLTrainer trainer.
        
        Args:
            agent: The RL agent containing the policy model
            tokenizer: Tokenizer for text processing
            accelerator: HuggingFace accelerator for distributed training
            lm_lr: Learning rate for the language model
            epochs: Number of training epochs per update
            max_grad_norm: Maximum gradient norm for clipping
            grad_accum_steps: Number of gradient accumulation steps
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        
        # Initialize optimizer for the policy model
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)
        
        # Training hyperparameters
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.step = 0
    
    def prepare(self):
        """Prepare the optimizer for distributed training using accelerator."""
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)

    def actor_loss(self, observation, image_features, action, **kwargs):
        """
        Calculate the behavioral cloning loss.
        
        Args:
            observation: Text observations from the environment
            image_features: Visual features from the environment
            action: Expert actions to imitate
            **kwargs: Additional arguments (ignored)
            
        Returns:
            dict: Dictionary containing the loss value
        """
        # Move image features to the correct device
        image_features = image_features.to(self.agent.device)
        
        # Calculate negative log probability as BC loss
        # This encourages the agent to assign high probability to expert actions
        log_probs = self.agent.get_log_prob(observation, image_features, action)
        loss = -log_probs.sum(dim=1).mean()
        
        # Backward pass using accelerator for distributed training
        self.accelerator.backward(loss)
        
        return {"bc.loss": loss.detach().cpu().item()}

    def update(self, replay_buffer, no_update_actor=False):
        """
        Perform one training update using data from the replay buffer.
        
        Args:
            replay_buffer: Buffer containing expert demonstrations
            no_update_actor: If True, skip actor updates (for warmup phases)
            
        Returns:
            dict: Training statistics and losses
        """
        self.step += 1
        info = {}
        info_list = []
        
        # Skip actor updates during warmup or if explicitly disabled
        if no_update_actor:
            return info
            
        # Determine batch size based on model type
        # LLaMA models typically require smaller batch sizes due to memory constraints
        action_bsize = 1 if 'llama' in self.agent.policy_lm else replay_buffer.batch_size
        
        # Train for multiple epochs
        for epoch in range(self.epochs):
            self.lm_optimizer.zero_grad()
            
            # Sample data for gradient accumulation
            # Total samples = grad_accum_steps * batch_size
            data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps * replay_buffer.batch_size)]
            
            # Flatten the sampled data structure
            for d in data:
                for k, v in d.items():
                    d[k] = v[0]  # Remove extra dimension from sampling
            
            # Create dataloader for batch processing
            dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)
            dataloader = self.accelerator.prepare(dataloader)
            
            # Process each batch and accumulate gradients
            for batch in dataloader:
                batch_info = self.actor_loss(**batch)
                info_list.append(batch_info)
            
            # Clip gradients to prevent exploding gradients
            self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            
            # Update model parameters
            self.lm_optimizer.step()
        
        # Calculate average training statistics
        info.update(dict_mean(info_list))
        return info

    def save(self, path):
        """
        Save the current training state including model weights and optimizer state.
        
        Args:
            path: Directory path to save the checkpoint
        """
        # Use accelerator's save_state for distributed training compatibility
        self.accelerator.save_state(path, safe_serialization=False)

    def load(self, path):
        """
        Load a previously saved training state.
        
        Args:
            path: Directory path containing the checkpoint
        """
        # Use accelerator's load_state for distributed training compatibility
        self.accelerator.load_state(path)