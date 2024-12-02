import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import imageio 
from IPython.display import Image
import matplotlib.pyplot as plt
# NormalizedActions wrapper
class NormalizedActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        self.action_space = gym.spaces.Box(low=np.full_like(low_bound, -1.), high=np.full_like(upper_bound, +1.))

    def action(self, action):
        low_bound = self.env.action_space.low
        upper_bound = self.env.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        return np.clip(action, low_bound, upper_bound)

    def reverse_action(self, action):
        low_bound = self.env.action_space.low
        upper_bound = self.env.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        return np.clip(action, -1.0, 1.0)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# Soft Actor-Critic components
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # Ensure both inputs are tensors
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)

        # Add batch dimension if necessary
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# Training SAC
def train_sac(env_name="HalfCheetah-v5", max_episodes=75, batch_size=256, gamma=0.99, tau=0.005):
    env = NormalizedActions(gym.make(env_name, render_mode="rgb_array"))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
    q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
    q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
    target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
    target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
    target_q_net1.load_state_dict(q_net1.state_dict())
    target_q_net2.load_state_dict(q_net2.state_dict())

    replay_buffer = ReplayBuffer(1000000)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
    q1_optimizer = optim.Adam(q_net1.parameters(), lr=3e-4)
    q2_optimizer = optim.Adam(q_net2.parameters(), lr=3e-4)

    def soft_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def compute_target(next_state, reward, done):
        with torch.no_grad():
            next_action = policy_net.get_action(next_state)
            target_q1 = target_q_net1(next_state, next_action)
            target_q2 = target_q_net2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            return reward + gamma * (1 - done) * target_q
    # Track the best candidate
    best_reward = float('-inf')
    best_policy_path = "best_policy.pth"
    episode_rewards = []  # List to store rewards for each episode


    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = policy_net.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
        
        
           # print(f"Step: {len(replay_buffer)}, Reward: {reward:.3f}, Done: {done}")

           # print(state, action, reward, terminated, truncated)


            if len(replay_buffer) > batch_size:

                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # Update Q Networks
                target = compute_target(next_states, rewards, dones)
                q1_loss = ((q_net1(states, actions) - target) ** 2).mean()
                q2_loss = ((q_net2(states, actions) - target) ** 2).mean()
                q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_optimizer.step()
                q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_optimizer.step()

                # Update Policy Network
                policy_loss = -q_net1(states, policy_net(states)).mean()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Update Target Networks
                soft_update(target_q_net1, q_net1)
                soft_update(target_q_net2, q_net2)

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
        # Save the best policy
        episode_rewards.append(episode_reward)  # Save the total reward for the episode

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(policy_net.state_dict(), best_policy_path)
            print(f"New best policy saved with reward: {best_reward:.3f}")


    # Save the policy and generate a GIF
    print("Training complete. Visualizing the trained policy...")

   
    policy_net.load_state_dict(torch.load(best_policy_path))
    gif_image = visualize(policy_net, env_name) 
    gif_image
# Plotting the reward function over episodes using matplotlib
    plt.plot(range(1, max_episodes + 1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Episodes')
    plt.show()
    #env.close()

# Visualization function for SAC
def visualize(policy_net, env_name="HalfCheetah-v5", max_steps=200, gif_path="...\\project\\sac_visualization_.gif"):
    env = NormalizedActions(gym.make(env_name, render_mode="rgb_array"))
    state, _ = env.reset()
    frames = []
    done = False
    steps = 0

    while not done and steps < max_steps:
        # Get action from the trained policy
        action = policy_net.get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Render and capture the frame
        
        frame = env.render()
        frames.append(frame)
        steps += 1

    env.close()

    # Save the frames as a GIF
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Visualization saved as {gif_path}")

    # Display the GIF if in Jupyter Notebook
    return Image(filename=gif_path)

# Run the training and visualization
if __name__ == "__main__":
    train_sac()
