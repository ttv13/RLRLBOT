import os
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
import torch
from rlgym import make
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.action_parsers.default_act import DefaultAction



class GoalReward(RewardFunction):
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


class ActionRepeatParser(DefaultAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_buffer = None
        self.action_repeat_counter = 0
        self.repeat_duration = 15

    def get_action(self, state):
        if self.action_repeat_counter == 0:
            self.action_buffer = super().get_action(state)
        self.action_repeat_counter += 1

        if self.action_repeat_counter >= self.repeat_duration:
            self.action_repeat_counter = 0

        return self.action_buffer

def make_env():
    def _init():
        return make(
            reward_fn=GoalReward(),
            obs_builder=DefaultObs(),
            action_parser=ActionRepeatParser(),
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
    model.learn(total_timesteps=1_000_000, callback=ProgressCallback(print_interval=10000))
    print("Training completed!")

    model.save("rocket_league_goal_agent")
    print("Model updated and saved as rocket_league_goal_agent.zip")

    env.close()
