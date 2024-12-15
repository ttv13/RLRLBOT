import os
import numpy as np
import random
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
import torch
from rlgym import make
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils import common_values
from rlgym.utils.action_parsers import ActionParser
from gym.spaces import Discrete
import gym

from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.reward_functions import RewardFunction


class obs_builder(ObsBuilder):
  def reset(self, initial_state: GameState):
    pass

  def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
    obs = []
    
    #If this observation is being built for a player on the orange team, we need to invert all the physics data we use.
    inverted = player.team_num == common_values.ORANGE_TEAM
    
    if inverted:
      obs += state.inverted_ball.serialize()
    else:
      obs += state.ball.serialize()
      
    for player in state.players:
      if inverted:
        obs += player.inverted_car_data.serialize()
      else:
        obs += player.car_data.serialize()
    
    return np.asarray(obs, dtype=np.float32)


class reward_fn(RewardFunction):
    def reset(self, initial_state):
        self.previous_ball_position = initial_state.ball.position
        self.previous_touch = False

    def get_reward(self, player, state, previous_action):
        goal_position = np.array([0, 5120, 0])
        ball_position = state.ball.position
        distance_to_goal = np.linalg.norm(goal_position - ball_position)

        reward = -distance_to_goal

        if np.linalg.norm(ball_position - self.previous_ball_position) > 10:
            reward += 10
            self.previous_touch = True
        else:
            reward -= 1
            self.previous_touch = False

        # Big reward for scoring a goal
        if ball_position[1] > 5100 and abs(ball_position[0]) < 1000:  # Close to goal
            reward += 500  # Large reward for scoring

        # Optionally, reward for getting closer to the goal (ball position near the goal line)
        if distance_to_goal < 1000:  # Ball is near the goal line
            reward += 50  # Reward for proximity to the goal

        self.previous_ball_position = ball_position
        return reward


class LookupAction(ActionParser):
    def __init__(self, bins=None):
        super().__init__()
        if bins is None:
            self.bins = [(-1, 0, 1)] * 5
        elif isinstance(bins[0], (float, int)):
            self.bins = [bins] * 5
        else:
            assert len(bins) == 5, "Need bins for throttle, steer, pitch, yaw and roll"
            self.bins = bins
        self._lookup_table = self.make_lookup_table(self.bins)

    @staticmethod
    def make_lookup_table(bins):
        actions = []
        # Ground
        for throttle in bins[0]:
            for steer in bins[1]:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in bins[2]:
            for yaw in bins[3]:
                for roll in bins[4]:
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        indexes = np.array(actions, dtype=np.int32)
        indexes = np.squeeze(indexes)
        return self._lookup_table[indexes]


def make_env():

    def _init():

        return make(
            reward_fn = reward_fn(),
            obs_builder= obs_builder(),
            action_parser=LookupAction(),
            terminal_conditions=[TimeoutCondition(300), GoalScoredCondition()],
            game_speed=100,

        )
    
    return _init

class ProgressCallback:
    def __init__(self, print_interval=10000):
        self.print_interval = print_interval

    def __call__(self, locals_, globals_):
        if locals_['self'].num_timesteps % self.print_interval == 0:
            print(f"Progress: {locals_['self'].num_timesteps} timesteps")
        return True

# Custom learning rate schedule
def lr_schedule_fn(current_timestep: int) -> float:
    initial_lr = 1e-3
    final_lr = 1e-5
    total_timesteps = 1_000_000
    lr = initial_lr + (final_lr - initial_lr) * (current_timestep / total_timesteps)
    return lr

if __name__ == "__main__":
    env = SubprocVecEnv([make_env() for _ in range(1)])
    env = VecMonitor(env)

    model_path = "rocket_league_goal_agent2.zip"
    if os.path.exists(model_path):
        print(f"Loading the model from {model_path}...")
        model = PPO.load(model_path, env=env)
    else:
        print("No saved model found. Initializing a new model...")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_rlgym_logs/",
            learning_rate=lr_schedule_fn,  # Custom learning rate schedule
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
        )

    set_random_seed(42)

    print("Training started...")
    model.learn(total_timesteps=100_000, callback=ProgressCallback(print_interval=1000))
    print("Training completed!")

    model.save("rocket_league_goal_agent")
    print("Model updated and saved as rocket_league_goal_agent.zip")

    env.close()
