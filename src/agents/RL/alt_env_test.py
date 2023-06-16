import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import FleetRL
from FleetRL.fleet_env.fleet_environment import FleetEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from FleetRL.fleet_env.alt_env import AltEnv

time_now = int(time.time())
comment = "alt_env_no_price"
trained_agents_lmd = f"./trained/last_mile_delivery/TD3-{time_now}-{comment}"
logs_lmd = f"./logs/last_mile_delivery/TD3-{time_now}-{comment}"

if not os.path.exists(trained_agents_lmd):
    os.makedirs(trained_agents_lmd)

if not os.path.exists(logs_lmd):
    os.makedirs(logs_lmd)

env = AltEnv(schedule_name="lmd_sched_single.csv",
             building_name="load_lmd.csv",
             include_building=True,
             include_pv=True,
             static_time_picker=False,
             deg_emp=False,
             include_price=True,
             ignore_price_reward=True,
             ignore_invalid_penalty=False,
             ignore_overcharging_penalty=False,
             ignore_overloading_penalty=False,
             episode_length=36,
             verbose=1,
             calculate_degradation=False,
             )

check_env(env)

env = DummyVecEnv([lambda: AltEnv(schedule_name="lmd_sched_single.csv",
                                  building_name="load_lmd.csv",
                                  include_building=True,
                                  include_pv=True,
                                  static_time_picker=False,
                                  deg_emp=False,
                                  include_price=True,
                                  ignore_price_reward=True,
                                  ignore_invalid_penalty=False,
                                  ignore_overcharging_penalty=False,
                                  ignore_overloading_penalty=False,
                                  episode_length=36,
                                  verbose=1,
                                  calculate_degradation=False,
                                  )])

env = VecCheckNan(env, raise_exception=True)

model = TD3('MlpPolicy', env)
model.learn(total_timesteps=int(2e5))

'''
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5)*np.ones(n_actions))

model = TD3('MlpPolicy',
            env,
            verbose=1,
            tensorboard_log="./FleetRl_tensorboard_single/",
            learning_rate=0.05,
            tau=0.003,
            gamma=0.95,
            learning_starts=10000,
            buffer_size=50000,
            batch_size=128)


saving_interval = 20000
for i in range(1, 10):
    model.learn(total_timesteps=saving_interval, reset_num_timesteps=False, tb_log_name=f"TD3_{time_now}_{comment}")
    model.save(f"{trained_agents_lmd}/{saving_interval * i}")
    
'''
