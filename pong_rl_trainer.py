import os
import time
import torch
import numpy as np
import subprocess
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from game_state import get_game_state
import pyautogui
import sys

# Define a simple neural network for the agent
class PongAgent(torch.nn.Module):
    def __init__(self):
        super(PongAgent, self).__init__()

        # Simple network with 3 inputs: ball_x, ball_y, paddle_y
        self.fc1 = torch.nn.Linear(3, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 3)  # 3 actions: stay, up, down

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)

class PongTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = PongAgent().to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.001)

        # Training parameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32

        # Experience replay buffer
        self.memory = []
        self.max_memory_size = 10000

        # Game parameters
        self.game_process = None
        self.ball_prev_y = None  # To track ball movement


    def start_game(self):
        """Start the Pong game"""
        # If game is already running, terminate it
        if self.game_process:
            self.game_process.terminate()
            time.sleep(1)

        # Start the game process
        # Update this path to point to your actual executable
        executable_path = os.path.join(os.getcwd(), "build/project_pong_cpp")

        # Check if executable exists
        if not os.path.exists(executable_path):
            print(f"Error: Executable not found at {executable_path}")
            print("Looking for executable...")

            # Search for the executable in the current directory structure
            for root, dirs, files in os.walk(os.getcwd()):
                if "project_pong_cpp" in files:
                    executable_path = os.path.join(root, "project_pong_cpp")
                    print(f"Found executable at: {executable_path}")
                    break

        # Ensure the file is executable
        if os.path.exists(executable_path):
            # Make the file executable
            os.chmod(executable_path, 0o755)
            print(f"Starting game at: {executable_path}")
            self.game_process = subprocess.Popen(executable_path)
        else:
            print(f"Error: Could not find the Pong executable. Make sure you've built the project.")
            sys.exit(1)

        # Wait for game to start
        time.sleep(2)


    # def start_game(self):
    #     """Start the Pong game"""
    #     # If game is already running, terminate it
    #     if self.game_process:
    #         self.game_process.terminate()
    #         time.sleep(1)
    #
    #     # Start the game process
    #     executable_path = os.path.join(os.getcwd(), "build/project_pong_cpp")
    #     self.game_process = subprocess.Popen(executable_path)
    #
    #     # Wait for game to start
    #     time.sleep(2)

    def get_state_vector(self, game_state):
        """Convert game state to a vector for the neural network"""
        if game_state['ball_pos'] is None:
            # If ball not found, use default position
            ball_x, ball_y = 0.5, 0.5
        else:
            ball_x, ball_y = game_state['ball_pos']

        # Use right paddle for training (assuming AI controls right paddle)
        paddle_y = game_state['right_paddle_pos']

        return np.array([ball_x, ball_y, paddle_y])

    def get_reward(self, game_state, prev_game_state=None):
        """Calculate reward based on game state"""
        reward = 0

        # If ball position isn't available, return 0 reward
        if game_state['ball_pos'] is None:
            return 0

        ball_x, ball_y = game_state['ball_pos']
        paddle_y = game_state['right_paddle_pos']

        # Small reward for keeping paddle aligned with ball
        proximity_reward = -abs(ball_y - paddle_y)
        reward += proximity_reward * 0.1

        # Big reward/penalty for scoring
        if prev_game_state and game_state['right_score'] > prev_game_state['right_score']:
            reward += 10  # We scored!
        if prev_game_state and game_state['left_score'] > prev_game_state['left_score']:
            reward -= 10  # They scored :(

        return reward

    def take_action(self, action):
        """Execute action in the game"""
        # Actions: 0 = do nothing, 1 = up, 2 = down
        if action == 1:  # Up
            pyautogui.keyDown('up')
            time.sleep(0.05)
            pyautogui.keyUp('up')
        elif action == 2:  # Down
            pyautogui.keyDown('down')
            time.sleep(0.05)
            pyautogui.keyUp('down')

        # Small wait to let the game update
        time.sleep(0.01)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def replay(self):
        """Train the agent on batches of experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in batch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

            # Get current Q value
            current_q = self.agent(state_tensor)[0][action]

            # Get target Q value
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * torch.max(self.agent(next_state_tensor)[0])

            # Calculate loss
            loss = torch.nn.functional.mse_loss(current_q.unsqueeze(0), torch.tensor([target_q]).to(self.device))

            # Update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(3)  # Random action

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.agent(state_tensor)[0]
        return q_values.argmax().item()  # Best action

    def train(self, episodes=100):
        """Train the agent for a specified number of episodes"""
        self.start_game()

        # Training metrics
        scores = []
        epsilons = []

        for episode in range(episodes):
            # Reset for new episode
            self.start_game()
            time.sleep(2)  # Wait for game to start

            prev_game_state = None
            total_reward = 0
            done = False

            # Episode loop
            step = 0
            max_steps = 1000  # Limit steps per episode

            while not done and step < max_steps:
                # Get current state
                screenshot = np.array(pyautogui.screenshot())
                game_state = get_game_state(screenshot)

                # Check if game is over
                if prev_game_state and (
                        game_state['left_score'] >= 10 or
                        game_state['right_score'] >= 10
                ):
                    done = True

                # Convert game state to vector
                state_vector = self.get_state_vector(game_state)

                # Select and perform action
                action = self.select_action(state_vector)
                self.take_action(action)

                # Get new state and reward
                new_screenshot = np.array(pyautogui.screenshot())
                new_game_state = get_game_state(new_screenshot)
                new_state_vector = self.get_state_vector(new_game_state)

                reward = self.get_reward(new_game_state, prev_game_state)
                total_reward += reward

                # Store in replay memory
                self.remember(state_vector, action, reward, new_state_vector, done)

                # Train the network
                self.replay()

                # Update previous state
                prev_game_state = new_game_state

                step += 1

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Record metrics
            scores.append(total_reward)
            epsilons.append(self.epsilon)

            print(f"Episode: {episode+1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")

            # Save model periodically
            if (episode + 1) % 10 == 0:
                torch.save(self.agent.state_dict(), f"pong_agent_episode_{episode+1}.pth")

        # Cleanup
        if self.game_process:
            self.game_process.terminate()

        # Plot results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(scores)
        plt.title('Score per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Score')

        plt.subplot(1, 2, 2)
        plt.plot(epsilons)
        plt.title('Epsilon per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')

        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()

        return scores

    def load_model(self, path):
        """Load a trained model"""
        self.agent.load_state_dict(torch.load(path))

    def play_game(self):
        """Let the trained agent play the game"""
        self.start_game()
        time.sleep(2)  # Wait for game to start

        # Set to evaluation mode
        self.agent.eval()
        self.epsilon = 0  # No exploration, just use the policy

        total_reward = 0
        done = False
        prev_game_state = None

        # Play until game is over
        step = 0
        max_steps = 5000  # Limit steps for safety

        print("Agent is now playing...")

        while not done and step < max_steps:
            # Get current state
            screenshot = np.array(pyautogui.screenshot())
            game_state = get_game_state(screenshot)

            # Check if game is over
            if prev_game_state and (
                    game_state['left_score'] >= 10 or
                    game_state['right_score'] >= 10
            ):
                done = True

            # Convert game state to vector
            state_vector = self.get_state_vector(game_state)

            # Select action
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.agent(state_tensor)[0]
            action = q_values.argmax().item()

            # Take action
            self.take_action(action)

            # Get reward
            new_screenshot = np.array(pyautogui.screenshot())
            new_game_state = get_game_state(new_screenshot)
            reward = self.get_reward(new_game_state, prev_game_state)
            total_reward += reward

            # Update previous state
            prev_game_state = new_game_state

            step += 1

            # Print progress occasionally
            if step % 100 == 0:
                print(f"Step {step}, Current score: {game_state['right_score']} - {game_state['left_score']}")

        print(f"Game over! Final score: {game_state['right_score']} - {game_state['left_score']}")
        print(f"Total reward: {total_reward}")

        # Cleanup
        if self.game_process:
            self.game_process.terminate()

# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train or play Pong with RL')
    parser.add_argument('mode', choices=['train', 'play'], help='Mode: train a new model or play with trained model')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes for training')
    parser.add_argument('--model_path', type=str, help='Path to trained model for play mode', default=None)

    args = parser.parse_args()

    trainer = PongTrainer()

    if args.mode == 'train':
        print(f"Training for {args.episodes} episodes...")
        trainer.train(episodes=args.episodes)
        print("Training completed!")
    elif args.mode == 'play':
        if args.model_path:
            print(f"Loading model from {args.model_path}...")
            trainer.load_model(args.model_path)
        else:
            print("No model path provided, using untrained model...")
        trainer.play_game()