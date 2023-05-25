import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
import time
import os

import FleetRL

time_now = int(time.time())
trained_agents_dir = f"./trained/vec_TD3-{time_now}"
logs_dir = f"./logs/vec_TD3-{time_now}"

if not os.path.exists(trained_agents_dir):
    os.makedirs(trained_agents_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

n_cpu = 4
vec_env = make_vec_env('FleetEnv-v0', n_envs=n_cpu)


# n_actions = vec_env.action_space.shape[-1]
# param_noise = None
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5)*np.ones(n_actions))

model = TD3("MlpPolicy", vec_env, verbose=1, train_freq=8, tensorboard_log="./FleetRl_tensorboard_1/", learning_starts=1000, learning_rate=0.0001)

model.learn(total_timesteps=100000, reset_num_timesteps=False, tb_log_name=f"vec_TD3_{time_now}")

# del model

# model = TD3.load(f"{trained_agents_dir}/{saving_interval * 4}", env=vec_env)
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
