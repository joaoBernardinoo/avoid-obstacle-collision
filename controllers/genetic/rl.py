# Copyright 1996-2024 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Optimized Webots controller for a navigation robot using reinforcement learning.
This script implements a Gymnasium environment and uses Stable-Baselines3
for training a PPO agent with multi-modal (Lidar, Camera) observations.
"""

# Standard libraries
import argparse
import numpy as np
import os
from dataclasses import dataclass, field
from pathlib import Path

# Webots libraries
from controller import Supervisor, Lidar, Camera, Motor, Keyboard

# RL libraries
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# --- Configuration ---


@dataclass
class Config:
    """Centralized configuration for the RL environment and training."""
    # Robot physics
    MAX_SPEED: float = 6.28
    FORWARD_SPEED_FACTOR: float = 0.4
    TURN_RATE_FORWARD: float = 0.5
    TURN_RATE_BACKWARD: float = 0.2

    # Environment settings
    LIDAR_SAMPLES: int = 20
    MAX_EPISODE_STEPS: int = 2048
    DISTANCE_THRESHOLD: float = 0.5  # Target reached
    COLLISION_THRESHOLD: float = 0.2  # Obstacle collision

    # Reward shaping
    REWARD_GOAL: float = 250.0
    PENALTY_COLLISION: float = -150.0
    PENALTY_STOP_ACTION: float = -20.0
    PENALTY_TIME_STEP: float = 0.0
    REWARD_DISTANCE_MULTIPLIER: float = 500.0

    # Training
    PPO_N_STEPS: int = 2048
    TOTAL_TIMESTEPS: int = 1_000_000
    DEVICE: str = 'cuda'
    MODEL_PATH: str = "ppo_navigation_model.zip"
    LOG_DIR: str = "./logs/"

    # Action mapping
    ACTION_MAP: dict = field(default_factory=lambda: {
        0: "forward",
        1: "turn_left",
        2: "turn_right",
        3: "stop"
    })

# --- Main Environment Class ---


class NavigationEnv(Supervisor, gym.Env):
    """
    Gymnasium environment for a robot navigating to a target using sensor data.
    The Supervisor is used only for world resets and reward calculation.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        self._timestep = int(self.getBasicTimeStep())

        # Setup Webots devices and nodes
        self._setup_webots_devices()

        # Setup Gymnasium spaces
        self._setup_gymnasium_spaces()

        # State variables
        self.current_step = 0
        self.last_distance_to_target = 0.0
        self.simulation_running = True

        self.__cache_observation = None

    def _setup_webots_devices(self):
        """Initializes Webots sensors, motors, and supervisor nodes."""
        self.lidar: Lidar = self.getDevice('Ibeo Lux')
        self.lidar.enable(self._timestep)
        self.lidar.enablePointCloud()

        self.camera: Camera = self.getDevice('camera')
        self.camera.enable(self._timestep)

        self.keyboard: Keyboard = self.getKeyboard()
        self.keyboard.enable(self._timestep)

        self.wheels: list[Motor] = [self.getDevice(name) for name in
                                    ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']]
        for wheel in self.wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)

        v = self.config.MAX_SPEED
        fwd_speed = v * self.config.FORWARD_SPEED_FACTOR
        turn_fwd = v * self.config.TURN_RATE_FORWARD
        turn_back = v * self.config.TURN_RATE_BACKWARD

        self.velocities = {
            "forward":    [fwd_speed, fwd_speed, fwd_speed, fwd_speed],
            "turn_left":  [turn_back, turn_fwd, turn_back, turn_fwd],
            "turn_right": [turn_fwd, turn_back, turn_fwd, turn_back],
            "stop":       [0.0, 0.0, 0.0, 0.0]
        }

        # Supervisor nodes
        self.robot_node = self.getSelf()
        self.target_node = self.getFromDef("TARGET")
        if not self.target_node:
            raise ValueError("Supervisor could not find DEF 'TARGET' node.")

        # Store initial positions for reset
        self.initial_robot_translation = self.robot_node.getField(
            'translation').getSFVec3f()
        self.initial_robot_rotation = self.robot_node.getField(
            'rotation').getSFRotation()

    def _setup_gymnasium_spaces(self):
        """Defines the observation and action spaces for the Gym environment."""
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=0, high=1, shape=(self.config.LIDAR_SAMPLES,), dtype=np.float32),
            'camera': spaces.Box(low=0, high=255, shape=(self.camera.getHeight(), self.camera.getWidth(), 3), dtype=np.uint8)
        })
        self.action_space = spaces.Discrete(len(self.config.ACTION_MAP))

    def reset(self, *, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)

        # Reset simulation and robot pose
        self.simulationResetPhysics()
        self.simulationReset()
        self.robot_node.getField('translation').setSFVec3f(
            self.initial_robot_translation)
        self.robot_node.getField('rotation').setSFRotation(
            self.initial_robot_rotation)
        super().step(self._timestep)

        # Re-initialize wheels
        for wheel in self.wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)

        self.current_step = 0
        self.last_distance_to_target = self._get_distance_to_target()
        self._apply_action(3)  # Stop action

        return self._get_observation(), {}

    def step(self, action):
        """Executes one time step within the environment."""
        if super().step(self._timestep) == -1:
            self.simulation_running = False
            return self.observation_space.sample(), 0, True, True, {}

        # Check for keyboard input to interrupt training or show current action
        key = self.keyboard.getKey()
        if key == ord('C'):
            raise KeyboardInterrupt  # Interrupt training when 'C' is pressed
        elif key == ord('S'):
            action_scalar = action.item() if isinstance(action, np.ndarray) else action
            action_name = self.config.ACTION_MAP.get(action_scalar, "unknown")
            print(f"Current Action: {action_name}")
            # Calculate and display current reward components for debugging
            dist_to_target = self._get_distance_to_target()
            distance_reward = (self.last_distance_to_target -
                               dist_to_target) * self.config.REWARD_DISTANCE_MULTIPLIER
            time_penalty = self.config.PENALTY_TIME_STEP
            total_reward = time_penalty + distance_reward
            print(
                f"Reward Components - Time Penalty: {time_penalty:.2f}, Distance Reward: {distance_reward:.2f}, Total (so far): {total_reward:.2f}")
            if dist_to_target < self.config.DISTANCE_THRESHOLD:
                print(
                    f"Goal Reward: {self.config.REWARD_GOAL:.2f} (would be added if terminated)")
            if np.min(self.lidar.getRangeImage()) < self.config.COLLISION_THRESHOLD:
                print(
                    f"Collision Penalty: {self.config.PENALTY_COLLISION:.2f} (would be added if terminated)")

        # The action from DummyVecEnv is a numpy array, get the scalar value
        action_scalar = action.item() if isinstance(action, np.ndarray) else action
        self._apply_action(action_scalar)
        self.current_step += 1

        obs = self._get_observation()
        reward, terminated = self._calculate_reward()
        truncated = self.current_step >= self.config.MAX_EPISODE_STEPS

        if terminated:
            print(f"✅ Success: Robot reached the target! Reward: {reward:.2f}")
        elif truncated:
            print(f"⌛ Truncated: Max episode steps reached.")

        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        """Gets and processes sensor data for the agent's observation."""
        # Lidar data processing

        if self.current_step % 3 == 0:
            self.__cache_observation = None

        if self.__cache_observation is not None:
            return self.__cache_observation

        lidar_raw = self.lidar.getRangeImage()
        max_range = self.lidar.getMaxRange()

        # Subsample and normalize lidar data
        indices = np.linspace(0, len(lidar_raw) - 1,
                              self.config.LIDAR_SAMPLES, dtype=int)
        lidar_obs = np.array(lidar_raw)[indices]
        lidar_obs[lidar_obs == np.inf] = max_range
        lidar_obs = np.clip(lidar_obs / max_range, 0, 1).astype(np.float32)

        # Camera data processing
        camera_image = np.frombuffer(self.camera.getImage(), np.uint8).reshape(
            (self.camera.getHeight(), self.camera.getWidth(), 4)
        )[:, :, :3]  # Get RGB from BGRA

        self.__cache_observation = {'lidar': lidar_obs, 'camera': camera_image}
        return self.__cache_observation

    def _apply_action(self, action_id):
        """Maps an action ID to robot motor velocities."""
        action_name = self.config.ACTION_MAP.get(action_id, "stop")

        target_velocities = self.velocities[action_name]
        for i, wheel in enumerate(self.wheels):
            wheel.setVelocity(target_velocities[i])

    def _calculate_reward(self):
        """Calculates the reward for the current state."""
        terminated = False
        reward = self.config.PENALTY_TIME_STEP

        dist_to_target = self._get_distance_to_target()
        min_lidar_dist = np.min(self.lidar.getRangeImage())

        # Reward for getting closer to the target
        reward += (self.last_distance_to_target - dist_to_target) * \
            self.config.REWARD_DISTANCE_MULTIPLIER

        # Goal reached
        if dist_to_target < self.config.DISTANCE_THRESHOLD:
            reward += self.config.REWARD_GOAL
            terminated = True
            print(
                f"✅ Success: Robot reached the target! Distance: {dist_to_target:.2f}")
        # Collision
        elif min_lidar_dist < self.config.COLLISION_THRESHOLD:
            reward += self.config.PENALTY_COLLISION
            terminated = True
            print(
                f"💥 Failure: Collision detected! Min distance: {min_lidar_dist:.2f}")

        self.last_distance_to_target = dist_to_target
        return reward, terminated

    def _get_distance_to_target(self):
        """Calculates the Euclidean distance to the target node."""
        robot_pos = np.array(self.robot_node.getPosition())
        target_pos = np.array(self.target_node.getPosition())
        return np.linalg.norm(robot_pos - target_pos)

# --- Main execution functions ---


def train_model(env, config: Config):
    """Trains the PPO model."""
    # Callbacks for evaluation and saving checkpoints.
    # NOTE: We pass the training 'env' to the callback. This is necessary because
    # Webots only allows one Robot instance per controller process. This will
    # raise a UserWarning from SB3, which can be safely ignored.
    eval_callback = EvalCallback(env, best_model_save_path=config.LOG_DIR,
                                 log_path=config.LOG_DIR,
                                 deterministic=False, render=False)

    checkpoint_callback = CheckpointCallback(save_freq=max(config.PPO_N_STEPS * 5, 1),
                                             save_path=config.LOG_DIR,
                                             name_prefix='ppo_nav_checkpoint')

    if os.path.exists(config.MODEL_PATH):
        print(f"Loading existing model from '{config.MODEL_PATH}'")
        model = PPO.load(config.MODEL_PATH, env=env, device=config.DEVICE)
    else:
        print("No model found. Creating a new PPO model.")
        model = PPO('MultiInputPolicy', env, n_steps=config.PPO_N_STEPS, verbose=1,
                    tensorboard_log=config.LOG_DIR, device=config.DEVICE)

    try:
        print(f"🚀 Starting training on {config.DEVICE.upper()}...")
        model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=[
            eval_callback, checkpoint_callback
        ],
            reset_num_timesteps=not os.path.exists(config.MODEL_PATH))
        model.save(config.MODEL_PATH)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        model.save(config.MODEL_PATH)
        print(f"Model saved to '{config.MODEL_PATH}'.")


def replay_simulation(env, config: Config):
    """Runs a replay of the trained model."""
    model_path = os.path.join(config.LOG_DIR, 'best_model.zip')
    if not os.path.exists(model_path):
        print(
            f"Error: Best model not found at '{model_path}'. Please train a model first.")
        return

    print(f"\n--- Starting Replay with best model: {model_path} ---")
    model = PPO.load(model_path, device=config.DEVICE)
    obs = env.reset()

    # The main loop for the replay
    # Access the underlying env to check the simulation_running flag
    while env.envs[0].env.simulation_running:
        action, _ = model.predict(obs, deterministic=True)
        step = env.step(action)
        obs, _, terminated, truncated = step

        if terminated[0] or truncated[0]:
            print("Episode finished. Resetting replay...")
            obs = env.reset()


def main():
    """Main function to parse arguments and run training or replay."""
    parser = argparse.ArgumentParser(
        description="Train or replay a PPO agent for robot navigation.")
    parser.add_argument("mode", choices=[
                        'train', 'replay'], help="Choose to 'train' a new model or 'replay' an existing one.")
    args = parser.parse_args()

    config = Config()
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Wrap the environment in a Monitor for proper logging, then in DummyVecEnv for SB3 compatibility.
    env = DummyVecEnv([lambda: Monitor(NavigationEnv(config))])

    try:
        if args.mode == 'train':
            train_model(env, config)
        elif args.mode == 'replay':
            replay_simulation(env, config)
    finally:
        # The environment should be closed properly
        print("Closing environment.")
        env.close()


if __name__ == '__main__':
    main()
