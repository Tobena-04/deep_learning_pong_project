import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from pong_env import PongEnv


# Define the neural network architecture
class PPONetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(6, 128)
        self.fc2   = nn.Linear(128, 128)
        self.actor = nn.Linear(128, 3)   # stay / up / down
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.actor(x), dim=1), self.critic(x)

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
        # state comes in as a 1‑D NumPy array (length 6)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 6)

        with torch.no_grad():
            action_probs, state_value = self.policy_old(state_t)

        dist   = Categorical(action_probs)
        action = dist.sample()              # tensor scalar

        if memory is not None:
            memory.states.append(state_t)   # keep batch dim for later cat()
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            memory.state_values.append(state_value.squeeze(0))

        return action.item()

    def update(self, memory):
        # Monte Carlo estimate of rewards
        rewards = []
        all_losses = []
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

            all_losses.append(loss.mean().item())

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        return np.mean(all_losses)

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

    # Tracking
    timestep = 0
    episode_rewards = []
    avg_rewards = []
    losses = []

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
                mean_loss = agent.update(memory)
                # old_len = len(losses)
                agent.update(memory)
                memory.clear_memory()
                losses.append(mean_loss)
                # Optional dummy loss value to align plot
                # losses.extend([None] * (len(episode_rewards) - old_len))

            # Update current state and reward
            state = next_state
            episode_reward += reward

        # Track episode reward
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)

        print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}, Avg (last 100): {avg_reward:.2f}")

        # Save the model
        if (episode + 1) % 50 == 0:
            torch.save(agent.policy.state_dict(), f"pong_policy_episode_{episode+1}.pth")

    env.close()

    # -------- PLOT RESULTS -------- #
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, label='Reward')
    plt.plot(avg_rewards, label='Avg Reward (100 episodes)')
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    if any(l is not None for l in losses):
        plt.subplot(2, 1, 2)
        plt.plot([l for l in losses if l is not None])
        plt.title('Loss (if captured)')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('ppo_training_results.png')
    plt.show()



if __name__ == "__main__":
    train_pong()