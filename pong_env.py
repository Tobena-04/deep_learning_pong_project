import cv2
import numpy as np
import pyautogui
import time
import gymnasium as gym
from gymnasium import spaces

class PongEnv(gym.Env):
    def __init__(self):
        super(PongEnv, self).__init__()
        # Action anad observation space
        self.action_space = spaces.Discrete(3)  # 0: do nothing, 1: up, 2: down

        # We'll use a preprocessed image as our state
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(80, 80, 1),
                                          dtype=np.float32)

        # Game parameters
        self.game_width = 1280
        self.game_height = 720
        self.max_score = 10
        self.start_game()

    # start the game by running the C++ scripts
    def start_game(self):
        """Start the C++ Pong game process"""
        import subprocess
        import os
        current_dir = os.getcwd()
        build_dir = os.path.join(current_dir, "project_pong_cpp/build")

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        os.chdir(build_dir)
        try:
            subprocess.run(["cmake", ".."], check=True)
            subprocess.run(["make"], check=True)
            self.game_process = subprocess.Popen("./project_pong_cpp")
            time.sleep(2)
        except subprocess.CalledProcessError as e:
            print(f"Error building: {e}")
        finally:
            os.chdir(current_dir)


    def get_screen(self):
        """Capture the game screen and process it"""
        # get screenshot with pyautogui
        screenshot = pyautogui.screenshot(region=(0, 0, self.game_width, self.game_height))

        # convert to grayscale (no other colors) array
        screen = np.array(screenshot)
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)



        # TODO: Probably a better way to do this with pytorch
        # Resize to 80x80
        resized = cv2.resize(gray, (80, 80))
        # Normalize
        processed_screen = resized / 255.0
        return processed_screen.reshape(80, 80, 1)

    def get_score(self):
        """Extract score from the screen using OCR or pixel-based detection"""
        # Placeholder for score detection
        # You could use OCR or look for specific pixel patterns


        # the score values are at the middle top of the screenshots
        return 0, 0

    def reset(self, **kwargs):
        """Reset the environment"""
        # Kill existing game process if any
        if hasattr(self, 'game_process'):
            self.game_process.terminate()
            time.sleep(1)

        # Start a new game
        self.start_game()

        # Get initial state
        observation = self.get_screen()

        return observation, {}

    def step(self, action):
        """Take an action in the environment"""
        # Execute action
        if action == 1:  # Move up
            pyautogui.keyDown('up')
            time.sleep(0.05)
            pyautogui.keyUp('up')
        elif action == 2:  # Move down
            pyautogui.keyDown('down')
            time.sleep(0.05)
            pyautogui.keyUp('down')

        # Wait for the game to update
        time.sleep(0.01)

        # Get new state
        next_state = self.get_screen()

        # Get current scores
        player_score, opponent_score = self.get_score()

        # Calculate reward
        reward = 0
        if player_score > self._prev_player_score:
            reward = 1
        elif opponent_score > self._prev_opponent_score:
            reward = -1

        # Update previous scores
        self._prev_player_score = player_score
        self._prev_opponent_score = opponent_score

        # Check if game is done
        done = player_score >= self.max_score or opponent_score >= self.max_score

        return next_state, reward, done, False, {}



    def close(self):
        """Clean up resources"""
        if hasattr(self, 'game_process'):
            self.game_process.terminate()