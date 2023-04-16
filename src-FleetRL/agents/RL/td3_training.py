import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import FleetRL

import gym

import os
import time

trained_agents_dir = f"trained/TD3-{int(time.time())}"
logs_dir = f"logs/TD3-{int(time.time())}"

if not os.path.exists(trained_agents_dir):
    os.makedirs(trained_agents_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env = gym.make('FleetEnv-v0')
check_env(env)
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5)*np.ones(n_actions))

model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=1)

saving_interval = 20000
for i in range(1,5):
    model.learn(total_timesteps=saving_interval, reset_num_timesteps=False, tb_log_name="TD3")
    model.save(f"{trained_agents_dir}/{saving_interval * i}")

env.close()
