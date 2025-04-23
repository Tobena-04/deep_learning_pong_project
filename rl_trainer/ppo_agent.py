import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import gymnasium as gym
from pong_env import PongEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- Memory Buffer ----------------------------- #
class Memory:
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.is_terminals = [], []

    def clear(self):
        self.__init__()


# -------------------------- Actor–Critic Network ------------------------- #
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hidden = 256
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x).squeeze(-1)
        return probs, value


# ------------------------------- PPO Agent ------------------------------- #
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=2.5e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    # ------------------------------------------------------------------ #
    def select_action(self, state, memory: Memory):
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            probs, _ = self.policy_old(state)
        probs = torch.clamp(probs, 1e-8, 1.0)  # avoid log(0)
        dist = Categorical(probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.log_probs.append(dist.log_prob(action))
        return action.item()

    # ------------------------------------------------------------------ #
    def update(self, memory: Memory):
        if len(memory.rewards) == 0:
            return np.nan

            # ------- Compute discounted returns ------- #
        returns = []
        discounted = 0
        for reward, is_term in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_term:
                discounted = 0
            discounted = reward + self.gamma * discounted
            returns.insert(0, discounted)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # ------- Convert memory to tensors ------- #
        states = torch.stack(memory.states).to(DEVICE)
        actions = torch.stack(memory.actions).to(DEVICE)
        old_log_probs = torch.stack(memory.log_probs).to(DEVICE)

        losses = []
        for _ in range(self.k_epochs):
            probs, state_values = self.policy(states)
            probs = torch.clamp(probs, 1e-8, 1.0)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratios = torch.exp(log_probs - old_log_probs.detach())
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() \
                   + 0.5 * self.mse_loss(state_values, returns) \
                   - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            losses.append(loss.item())

        self.policy_old.load_state_dict(self.policy.state_dict())
        return float(np.mean(losses))


# ------------------------- Environment Selection ------------------------- #

def make_env(env_name: str):
    if env_name == "custom":
        if PongEnv is None:
            raise RuntimeError("pong_env.py not importable – ensure it exists on PYTHONPATH")
        return PongEnv()
    return gym.make(env_name)


# ----------------------------- Training Loop ----------------------------- #

def train(cfg):
    env = make_env(cfg.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim,
                     lr=cfg.lr, gamma=cfg.gamma, eps_clip=cfg.eps_clip)
    memory = Memory()

    rewards_history, avg_rewards_history = [], []
    losses_history, avg_losses_history = [], []

    timestep = 0
    last_loss_value = None

    for ep in range(1, cfg.episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        # --------------- play one episode ---------------
        while not done:
            timestep += 1
            action = agent.select_action(state, memory)
            next_state, r, done, _, _ = env.step(action)

            memory.rewards.append(r)
            memory.is_terminals.append(done)
            state = next_state
            ep_reward += r

            # ---- update on schedule ----
            if timestep % cfg.update_timestep == 0:
                last_loss_value = agent.update(memory)
                memory.clear()
                losses_history.append(last_loss_value)

        # ---------- force one final update (per-episode) ----------
        if len(memory.rewards) > 0:
            last_loss_value = agent.update(memory)
            memory.clear()
            losses_history.append(last_loss_value)

        # ---------- record stats ----------
        rewards_history.append(ep_reward)
        avg_rewards_history.append(np.mean(rewards_history[-100:]))

        # avoid nan when losses_history is still empty
        if losses_history:
            avg_losses_history.append(np.mean(losses_history[-10:]))
        else:
            avg_losses_history.append(np.nan)

        # ---------- console log ----------
        avg_reward_str = f"{avg_rewards_history[-1]:7.2f}"
        loss_str = (
            f"{last_loss_value:8.4f}"
            if last_loss_value is not None and not np.isnan(last_loss_value)
            else "   --  "
        )
        print(
            f"Ep {ep:4d}/{cfg.episodes} | "
            f"Reward {ep_reward:8.2f} | "
            f"Avg(100) {avg_reward_str} | "
            f"Loss {loss_str}"
        )

    env.close()

    # ---------- draw curves ----------
    plot_curves(rewards_history, avg_rewards_history,
                losses_history, avg_losses_history)


# ---------------------------- Plotting Helper ---------------------------- #

def plot_curves(rewards, avg_rewards, losses, avg_losses):
    if len(losses) == 0:
        print("No loss values recorded – try lowering --update_timestep.")
        return

    plt.figure(figsize=(12, 8))

    # ---------- rewards ----------
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label="Reward")
    plt.plot(avg_rewards, label="Avg Reward (100 eps)")
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    # ---------- losses ----------
    plt.subplot(2, 1, 2)
    plt.plot(losses, label="Loss")
    plt.plot(avg_losses, label="Avg Loss (last 10)")
    plt.title("Loss per Update Step")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    out_path = "ppo_training_results.png"
    plt.savefig(out_path)
    print(f"Saved curves to {out_path}")
    plt.show()


# ----------------------------------- CLI --------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Start training")
    tr.add_argument("--env", type=str, default="custom", help="Gym env id or 'custom'")
    tr.add_argument("--episodes", type=int, default=1000)
    tr.add_argument("--lr", type=float, default=2.5e-4)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--eps_clip", type=float, default=0.2)
    tr.add_argument("--update_timestep", type=int, default=1024, help="Policy update interval")

    cfg = parser.parse_args()
    if cfg.cmd == "train":
        train(cfg)


if __name__ == "__main__":
    main()
