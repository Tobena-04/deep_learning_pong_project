import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import subprocess
import random

# Neural network for the agent
class PongAgent(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=3):
        super(PongAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)

# Memory buffer for storing experiences
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class PongRLTrainer:
    def __init__(self):
        # Files for communication with the C++ game
        self.state_file = "game_state.txt"
        self.action_file = "agent_action.txt"
        self.done_file = "game_done.txt"

        # Game process
        self.game_process = None

        # Training parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = PongAgent().to(self.device)
        self.target_agent = PongAgent().to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())

        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.0005)
        self.memory = ReplayBuffer(capacity=100000)

        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update = 10  # Update target network every N episodes

    def start_game(self):
        """Start the Pong game in RL mode"""
        # If game is already running, terminate it
        if self.game_process:
            self.game_process.terminate()
            time.sleep(1)

        # Find the path to the executable
        executable_path = None
        for root, dirs, files in os.walk(os.getcwd()):
            if "project_pong_cpp" in files:
                executable_path = os.path.join(root, "project_pong_cpp")
                print(f"Found executable at: {executable_path}")
                break

        if not executable_path:
            print("Error: Could not find the Pong executable!")
            return False

        # Make sure it's executable
        os.chmod(executable_path, 0o755)

        # Start the game with RL flag
        print(f"Starting game at: {executable_path}")
        self.game_process = subprocess.Popen([executable_path, "--rl"])
        time.sleep(2)  # Give it time to start

        return True

    def get_game_state(self):
        """Read the game state from the file"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                with open(self.state_file, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        values = line.split(',')
                        if len(values) == 8:
                            state = {
                                'ball_x': float(values[0]),
                                'ball_y': float(values[1]),
                                'ball_vel_x': float(values[2]),
                                'ball_vel_y': float(values[3]),
                                'left_paddle_y': float(values[4]),
                                'right_paddle_y': float(values[5]),
                                'left_score': int(values[6]),
                                'right_score': int(values[7])
                            }
                            return state
                    # If we get here, the file exists but doesn't have valid data
                    time.sleep(0.1)
            except (FileNotFoundError, ValueError):
                time.sleep(0.1)

        # If we've tried several times and failed, return None
        return None

    def set_agent_action(self, action):
        """Write the agent's action to the file"""
        with open(self.action_file, 'w') as f:
            f.write(str(action))

    def check_game_done(self):
        """Check if the game is done"""
        try:
            with open(self.done_file, 'r') as f:
                return f.readline().strip() == "1"
        except FileNotFoundError:
            return False

    def get_state_tensor(self, state):
        """Convert state dict to tensor for the neural network"""
        if state is None:
            return torch.zeros(1, 6).to(self.device)

        # Create state vector (excluding scores)
        state_vector = [
            state['ball_x'],
            state['ball_y'],
            state['ball_vel_x'],
            state['ball_vel_y'],
            state['left_paddle_y'],
            state['right_paddle_y']
        ]
        return torch.FloatTensor([state_vector]).to(self.device)

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return random.randrange(3)  # Random action

        state_tensor = self.get_state_tensor(state)
        with torch.no_grad():
            action_probs = self.agent(state_tensor)
        return torch.argmax(action_probs, dim=1).item()

    def calculate_reward(self, state, prev_state, done):
        """Calculate reward based on game state"""
        if state is None or prev_state is None:
            return 0

        reward = 0

        # Reward for scoring
        if state['right_score'] > prev_state['right_score']:
            reward += 10  # Big reward for scoring

        # Penalty for being scored against
        if state['left_score'] > prev_state['left_score']:
            reward -= 10

        # Small reward for keeping paddle aligned with ball
        ball_y = state['ball_y']
        paddle_y = state['right_paddle_y']
        alignment_reward = -abs(ball_y - paddle_y)
        reward += alignment_reward * 0.1

        # Extra reward for hitting the ball when it's moving toward us
        if state['ball_vel_x'] < 0 and prev_state['ball_vel_x'] > 0:
            reward += 1  # Successfully hit the ball

        return reward

    def train_batch(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.agent(state_batch).gather(1, action_batch)

        # Next Q values (from target network)
        next_q_values = self.target_agent(next_state_batch).max(1, keepdim=True)[0]
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Loss
        loss = nn.MSELoss()(q_values, expected_q_values.detach())

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        for param in self.agent.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train(self, episodes=1000, max_steps_per_episode=5000):
        """Train the agent for a number of episodes"""
        if not self.start_game():
            print("Failed to start game. Aborting training.")
            return

        rewards_history = []
        avg_rewards_history = []
        losses = []

        for episode in range(episodes):
            # Wait a bit for the game to reset
            time.sleep(1)

            # Reset tracking variables
            episode_reward = 0
            step = 0
            prev_state = None
            prev_action = None

            # Check if game has started properly
            state = self.get_game_state()
            if state is None:
                print("Cannot get game state. Make sure the game is running correctly.")
                break

            # Store initial scores
            prev_left_score = state['left_score']
            prev_right_score = state['right_score']

            # Main training loop
            done = False
            while not done and step < max_steps_per_episode:
                # Select and execute action
                action = self.select_action(state)
                self.set_agent_action(action)

                # Small delay to let the game update
                time.sleep(0.01)

                # Get new state
                new_state = self.get_game_state()

                # Check if game is done (someone reached 10 points)
                done = self.check_game_done()

                # Calculate reward
                reward = self.calculate_reward(new_state, state, done)
                episode_reward += reward

                # Store experience in memory
                if prev_state is not None:
                    state_vector = self.get_state_tensor(state).squeeze(0).cpu().numpy()
                    next_state_vector = self.get_state_tensor(new_state).squeeze(0).cpu().numpy()
                    self.memory.push(state_vector, action, reward, next_state_vector, done)

                # Train on a batch
                if len(self.memory) >= self.batch_size:
                    loss = self.train_batch()
                    if loss:
                        losses.append(loss)

                # Move to next state
                prev_state = state
                state = new_state
                step += 1

                # Print progress occasionally
                if step % 100 == 0:
                    score_diff = state['right_score'] - prev_right_score
                    prev_right_score = state['right_score']
                    print(f"Episode {episode+1}, Step {step}, Score: {state['right_score']}-{state['left_score']}, " +
                          f"Points in last 100 steps: {score_diff}, Reward: {episode_reward:.2f}")

            # End of episode
            rewards_history.append(episode_reward)
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            avg_rewards_history.append(avg_reward)

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Update target network
            if episode % self.target_update == 0:
                self.target_agent.load_state_dict(self.agent.state_dict())

            # Save model periodically
            if (episode + 1) % 20 == 0:
                torch.save(self.agent.state_dict(), f"pong_agent_episode_{episode+1}.pth")

            print(f"Episode {episode+1}/{episodes} completed. Total reward: {episode_reward:.2f}, " +
                  f"Average reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")

        # Training complete, save final model
        torch.save(self.agent.state_dict(), "pong_agent_final.pth")

        # Plot results
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(rewards_history)
        plt.plot(avg_rewards_history)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(['Reward', 'Average Reward (100 episodes)'])

        if losses:
            plt.subplot(2, 1, 2)
            plt.plot(losses)
            plt.title('Loss per Training Step')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')

        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()

        # Clean up
        if self.game_process:
            self.game_process.terminate()

    def load_model(self, model_path):
        """Load a trained model"""
        self.agent.load_state_dict(torch.load(model_path))

    def play(self, max_steps=5000):
        """Let a trained agent play the game"""
        if not self.start_game():
            print("Failed to start game. Aborting.")
            return

        # Set to evaluation mode
        self.agent.eval()
        self.epsilon = 0  # No exploration

        print("Agent is playing...")

        step = 0
        total_reward = 0
        prev_state = None

        # Start playing
        while step < max_steps:
            # Get current state
            state = self.get_game_state()

            if state is None:
                print("Cannot get game state")
                time.sleep(0.1)
                continue

            # Select action (no exploration)
            state_tensor = self.get_state_tensor(state)
            with torch.no_grad():
                action_probs = self.agent(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()

            # Execute action
            self.set_agent_action(action)

            # Wait a bit
            time.sleep(0.01)

            # Get reward for display purposes
            if prev_state:
                reward = self.calculate_reward(state, prev_state, False)
                total_reward += reward

            prev_state = state
            step += 1

            # Display score occasionally
            if step % 100 == 0:
                print(f"Step {step}, Score: {state['right_score']}-{state['left_score']}, Reward: {total_reward:.2f}")

            # Check if game is done
            if self.check_game_done():
                print("Game over!")
                break

        print(f"Final score: {state['right_score']}-{state['left_score']}")
        print(f"Total reward: {total_reward:.2f}")

        # Clean up
        if self.game_process:
            self.game_process.terminate()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train or play Pong with RL')
    parser.add_argument('mode', choices=['train', 'play'], help='Mode: train a new model or play with trained model')
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes for training')
    parser.add_argument('--model', type=str, help='Path to a trained model for play mode')

    args = parser.parse_args()

    trainer = PongRLTrainer()

    if args.mode == 'train':
        print(f"Training for {args.episodes} episodes...")
        trainer.train(episodes=args.episodes)
    elif args.mode == 'play':
        if args.model:
            print(f"Loading model from {args.model}...")
            trainer.load_model(args.model)
        else:
            print("No model specified, using untrained model...")
        trainer.play()