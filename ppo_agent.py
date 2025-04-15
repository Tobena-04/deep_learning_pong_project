import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

# Define the neural network architecture
class PPONetwork(nn.Module):
    def __init__(self):
        super(PPONetwork, self).__init__()

        # CNN layers to process game screen
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)

        # Actor (policy) and critic (value) heads
        self.actor = nn.Linear(512, 3)  # 3 actions: no-op, up, down
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        # Reshape input if it's not in the right format
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x  # Already in correct format (B, C, H, W)
        else:
            x = x.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)

        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply FC layer
        x = F.relu(self.fc1(x))

        # Get policy and value
        policy = F.softmax(self.actor(x), dim=1)
        value = self.critic(x)

        return policy, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PPONetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = PPONetwork().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory=None):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_probs, state_value = self.policy_old(state)

        # Sample action from the probability distribution
        dist = Categorical(action_probs)
        action = dist.sample()

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            memory.state_values.append(state_value)

        return action.item()

    def update(self, memory):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert lists to tensors
        old_states = torch.cat(memory.states).to(self.device).detach()
        old_actions = torch.cat(memory.actions).to(self.device).detach()
        old_logprobs = torch.cat(memory.logprobs).to(self.device).detach()
        old_state_values = torch.cat(memory.state_values).to(self.device).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            state_values = state_values.squeeze()

            # Compute ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Compute advantage
            advantages = rewards - old_state_values.detach().squeeze()

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist.entropy()

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

# Memory class for storing transitions
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:]

# Training function
def train_pong():
    # Create environment
    env = PongEnv()

    # Create agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape,
        action_dim=env.action_space.n
    )

    # Training parameters
    max_episodes = 1000
    update_timestep = 2000

    # Memory
    memory = Memory()

    # Training loop
    timestep = 0
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            timestep += 1

            # Select action
            action = agent.select_action(state, memory)

            # Take action
            next_state, reward, done, _, _ = env.step(action)

            # Save in memory
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # Update if its time
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()

            # Update current state and reward
            state = next_state
            episode_reward += reward

        print(f"Episode: {episode}, Reward: {episode_reward}")

        # Save the model
        if episode % 50 == 0:
            torch.save(agent.policy.state_dict(), f"pong_policy_episode_{episode}.pth")

    env.close()

if __name__ == "__main__":
    train_pong()