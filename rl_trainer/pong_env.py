import gymnasium as gym
import numpy as np
import os, time, subprocess

class PongEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, exe_path=None, max_steps=1000):
        super().__init__()
        self.exe_path = exe_path or self._find_exe()
        self.max_steps = max_steps
        # 6‑float state vector from your C++ side
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        # 0 = stay, 1 = up, 2 = down
        self.action_space = gym.spaces.Discrete(3)
        self.state_file = "game_state.txt"
        self.action_file = "agent_action.txt"
        self.done_file = "game_done.txt"
        self.proc = None
        self.step_count = 0

    def _find_exe(self):
        """
        Walk **upward** from CWD looking for */project_pong_cpp/build/project_pong_cpp*
        and, failing that, do a full recursive walk downward.
        """
        # 1) Walk upward a few levels (git‑style)
        cur = os.getcwd()
        for _ in range(4):                              # go up max 4 directories
            candidate = os.path.join(
                cur, "project_pong_cpp", "build", "project_pong_cpp"
            )
            if os.path.isfile(candidate):
                return candidate
            cur = os.path.dirname(cur)

        # 2) Fall back to a deep downward search
        for root, _, files in os.walk(os.getcwd()):
            if "project_pong_cpp" in files:
                return os.path.join(root, "project_pong_cpp")

        raise FileNotFoundError(
            "Could not locate project_pong_cpp executable.\n"
            "Try passing exe_path='./path/to/project_pong_cpp' when you create PongEnv."
        )


    # ---------- Gym API ---------- #
    def reset(self, seed=None, options=None):
        if self.proc: self.proc.terminate()
        os.chmod(self.exe_path, 0o755)
        self.proc = subprocess.Popen([self.exe_path, "--rl"])
        time.sleep(1)  # let the window open
        self.step_count = 0
        state = self._read_state()
        return state, {}

    def step(self, action):
        self._write_action(int(action))
        time.sleep(0.01)
        state = self._read_state()
        reward = self._calc_reward(state)
        done = self._check_done() or self.step_count >= self.max_steps
        self.step_count += 1
        return state, reward, done, False, {}

    def close(self):
        if self.proc: self.proc.terminate()

    # ---------- helper I/O ---------- #
    def _read_state(self, max_retries=10, delay=0.05):
        for _ in range(max_retries):
            try:
                with open(self.state_file, "r") as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split(",")
                        if len(parts) >= 6:
                            return np.array([float(x) for x in parts[:6]], dtype=np.float32)
            except Exception:
                pass
            time.sleep(delay)
        raise ValueError("Failed to read valid game state after retries")

    def _write_action(self, a):
        with open(self.action_file, "w") as f:
            f.write(str(a))

    def _check_done(self):
        try:
            with open(self.done_file, "r") as f:
                return f.readline().strip() == "1"
        except FileNotFoundError:
            return False

    def _calc_reward(self, s):
        # identical to direct_rl_trainer but pure‑python
        ball_y, paddle_y = s[1], s[5]
        align = -abs(ball_y - paddle_y) * 0.1
        return align  # leave scoring reward to the C++ side if you emit it
