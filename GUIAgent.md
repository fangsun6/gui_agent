1. DummyDataset
A simple PyTorch Dataset wrapper for replay buffer data.

Purpose: Converts list-based data into PyTorch Dataset format for DataLoader compatibility.

Usage:

2. ReplayBuffer
A circular buffer for storing and sampling expert demonstrations with multimodal support.

Key Features
Multimodal Storage: Handles text observations and visual features
Circular Buffer: Efficiently manages memory with fixed capacity
Monte Carlo Returns: Stores computed trajectory returns
Batch Sampling: Random sampling for training

Data Structure

ReplayBuffer stores:
├── observations          # Text descriptions of states
├── actions              # Expert actions (text/tokens)
├── image_features       # Visual features from current state
├── next_image_features  # Visual features from next state
├── rewards              # Step rewards
├── next_observations    # Next state text descriptions
├── dones               # Episode termination flags
└── mc_returns          # Monte Carlo returns

API Reference
ReplayBuffer.init(batch_size=2, capacity=10000)
Initialize the replay buffer.

Parameters:

batch_size (int): Default batch size for sampling
capacity (int): Maximum number of transitions to store
ReplayBuffer.insert(observation, action, image_features, ...)
Insert a single transition into the buffer.

Parameters:

observation (str): Text description of current state
action (str): Expert action taken
image_features (np.ndarray): Visual features from current state
next_image_features (np.ndarray): Visual features from next state
reward (float/np.ndarray): Step reward
next_observation (str): Text description of next state
done (bool/np.ndarray): Episode termination flag
mc_return (float/np.ndarray): Monte Carlo return

Example:
buffer = ReplayBuffer(batch_size=32, capacity=100000)

buffer.insert(
    observation="Click on the login button",
    action="click(100, 200)",
    image_features=screen_features,
    next_image_features=next_screen_features,
    reward=1.0,
    next_observation="Login dialog appeared",
    done=False,
    mc_return=5.2
)


Usage in Training Pipeline
1. Data Collection

# Initialize buffer
replay_buffer = ReplayBuffer(batch_size=32, capacity=500000)

# Insert expert demonstrations
for trajectory in expert_trajectories:
    for transition in trajectory:
        replay_buffer.insert(**transition)

2. Training Data Sampling

# Sample for training
batch = replay_buffer.sample(batch_size=16)

# Convert to PyTorch Dataset
dataset = DummyDataset([batch])
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Use in training loop
for batch_data in dataloader:
    loss = compute_bc_loss(batch_data)


Memory Management

Circular Buffer Design
Fixed Memory: Pre-allocates arrays for maximum efficiency
Overwrite Policy: Oldest data is overwritten when buffer is full
Type Safety: Proper dtype handling for mixed data types

Data Types Handled
Text Data: Stored as Python objects (variable length strings)
Visual Features: Fixed-size numpy arrays
Scalar Values: Rewards, returns, done flags
Mixed Precision: Automatic dtype preservation

Integration with Behavioral Cloning

# 1. Load expert trajectories
trajectories = torch.load("expert_demos.pt")

# 2. Create replay buffer
buffer = ReplayBuffer(batch_size=32, capacity=100000)

# 3. Insert all transitions
for traj in trajectories:
    for transition in traj:
        buffer.insert(**transition)

# 4. Sample for training
training_data = buffer.sample(batch_size=64)

# 5. Train BC model
loss = bc_trainer.actor_loss(**training_data)


Error Handling

Input Validation

# Automatic type conversion
if isinstance(reward, (float, int)):
    reward = np.array(reward)

# Shape validation
assert reward.shape == ()
assert done.shape == ()

