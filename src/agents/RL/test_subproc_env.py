import FleetRL
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import ProgressBarCallback

import numpy as np
from stable_baselines3 import TD3

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
import time
import os
from pink import PinkNoiseDist, PinkActionNoise
import multiprocessing
import FleetRL
from FleetRL.fleet_env.fleet_environment import FleetEnv

import FleetRL
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import ProgressBarCallback

import numpy as np
from stable_baselines3 import TD3

#
# def make_env(env_id: str, rank: int, seed: int = 0):
#     def _init():
#         env = gym.make(env_id, schedule_name="lmd_sched_single.csv",
#                        building_name="load_lmd.csv",
#                        include_building=False,
#                        include_pv=False,
#                        include_price=True,
#                        static_time_picker=False,
#                        target_soc=0.85,
#                        init_soh=1.0,
#                        deg_emp=False,
#                        ignore_price_reward=False,
#                        ignore_overloading_penalty=False,
#                        ignore_invalid_penalty=False,
#                        ignore_overcharging_penalty=False,
#                        episode_length=24,
#                        log_to_csv=False,
#                        calculate_degradation=False,
#                        verbose=1,
#                        normalize_in_env=False)
#
#         env.reset(seed=seed + rank)
#         return env
#
#     set_random_seed(seed)
#     return _init
#
#
# if __name__ == "__main__":
#     # env_id = "FleetEnv-v0"
#     # num_cpu = 4
#     # vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
#     #
#     # vec_env = VecNormalize(venv=vec_env, training=True, norm_obs=True, norm_reward=True)
#     #
#     # model = TD3("MlpPolicy", vec_env, verbose=1, train_freq=4)
#     # model.learn(total_timesteps=250)
#     #
#     # obs = vec_env.reset()
#     #
#     # for _ in range(1000):
#     #     action, _states = model.predict(obs)
#     #     obs, rewards, dones, info = vec_env.step(action)
#     #     print(rewards)
#     #     vec_env.render()
#
#     training_env = make_vec_env(env_id="FleetEnv-v0",
#                                 vec_env_cls=SubprocVecEnv,
#                                 n_envs=1,
#                                 env_kwargs={
#                                     "schedule_name": "lmd_sched_single.csv",
#                                     "building_name": "load_lmd.csv",
#                                     "include_building": False,
#                                     "include_pv": False,
#                                     "static_time_picker": False,
#                                     "deg_emp": False,
#                                     "include_price": False,
#                                     "ignore_price_reward": True,
#                                     "ignore_invalid_penalty": False,
#                                     "ignore_overcharging_penalty": False,
#                                     "ignore_overloading_penalty": True,
#                                     "episode_length": 36,
#                                     "verbose": 0,
#                                     "calculate_degradation": False
#                                 })
#
#     vec_train_env = VecNormalize(venv=training_env, training=True, norm_obs=True, norm_reward=True)
#
#     eval_env = make_vec_env(env_id="FleetEnv-v0",
#                             vec_env_cls=SubprocVecEnv,
#                             n_envs=1,
#                             env_kwargs={
#                                 "schedule_name": "lmd_sched_single.csv",
#                                 "building_name": "load_lmd.csv",
#                                 "include_building": False,
#                                 "include_pv": False,
#                                 "eval_time_picker": True,
#                                 "deg_emp": False,
#                                 "include_price": False,
#                                 "ignore_price_reward": True,
#                                 "ignore_invalid_penalty": False,
#                                 "ignore_overcharging_penalty": False,
#                                 "ignore_overloading_penalty": True,
#                                 "episode_length": 36,
#                                 "verbose": 0,
#                                 "calculate_degradation": False
#                             })
#
#     vec_eval_env = VecNormalize(venv=eval_env, training=True, norm_obs=True, norm_reward=True)
#
#     eval_callback = EvalCallback(vec_eval_env, best_model_save_path="./test_ev", log_path="./test_ev",
#                                  eval_freq=500, deterministic=True, render=False, verbose=1, warn=True, n_eval_episodes=5)
#
#     model = TD3("MlpPolicy", vec_train_env, verbose=0, train_freq=2)
#
#     print(model.policy)
#
#     #model.learn(total_timesteps=5000, callback=eval_callback, progress_bar=True)
#
#     # # Don't forget to save the VecNormalize statistics when saving the agent
#     # log_dir = "/tmp/"
#     # model.save(log_dir + "ppo_halfcheetah")
#     # stats_path = os.path.join(log_dir, "vec_normalize.pkl")
#     # env.save(stats_path)
#     #
#     # # To demonstrate loading
#     # del model, vec_env
#     #
#     # # Load the saved statistics
#     # vec_env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
#     # vec_env = VecNormalize.load(stats_path, vec_env)
#     # #  do not update them at test time
#     # vec_env.training = False
#     # # reward normalization is not needed at test time
#     # vec_env.norm_reward = False
#     #
#     # # Load the agent
#     # model = PPO.load(log_dir + "ppo_halfcheetah", env=vec_env)


if __name__ == "__main__":
    # env_id = "FleetEnv-v0"
    # num_cpu = 4
    # vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    #
    # vec_env = VecNormalize(venv=vec_env, training=True, norm_obs=True, norm_reward=True)
    #
    # model = TD3("MlpPolicy", vec_env, verbose=1, train_freq=4)
    # model.learn(total_timesteps=250)
    #
    # obs = vec_env.reset()
    #
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     print(rewards)
    #     vec_env.render()

    time_now = int(time.time())
    comment = "td3_full_no_ocinv"
    trained_agents_dir = f"./trained/TD3-{time_now}-{comment}"
    logs_dir = f"./logs/TD3-{time_now}-{comment}"

    if not os.path.exists(trained_agents_dir):
        os.makedirs(trained_agents_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    training_env = make_vec_env(FleetEnv,
                                vec_env_cls=SubprocVecEnv,
                                n_envs=4,
                                env_kwargs={
                                    "schedule_name": "lmd_sched_single.csv",
                                    "building_name": "load_lmd.csv",
                                    "include_building": True,
                                    "include_pv": True,
                                    "static_time_picker": False,
                                    "deg_emp": False,
                                    "include_price": True,
                                    "ignore_price_reward": False,
                                    "ignore_invalid_penalty": True,
                                    "ignore_overcharging_penalty": True,
                                    "ignore_overloading_penalty": False,
                                    "episode_length": 48,
                                    "verbose": 0,
                                    "calculate_degradation": False,
                                    "normalize_in_env": False
                                })

    vec_train_env = VecNormalize(venv=training_env, training=True, norm_obs=True, norm_reward=True)

    eval_env = make_vec_env(FleetEnv,
                            vec_env_cls=SubprocVecEnv,
                            n_envs=1,
                            env_kwargs={
                                "schedule_name": "lmd_sched_single.csv",
                                "building_name": "load_lmd.csv",
                                "include_building": True,
                                "include_pv": True,
                                "eval_time_picker": True,
                                "deg_emp": False,
                                "include_price": True,
                                "ignore_price_reward": False,
                                "ignore_invalid_penalty": True,
                                "ignore_overcharging_penalty": True,
                                "ignore_overloading_penalty": False,
                                "episode_length": 48,
                                "verbose": 0,
                                "calculate_degradation": False,
                                "normalize_in_env": False
                            })

    vec_eval_env = VecNormalize(venv=eval_env, training=True, norm_obs=True, norm_reward=True)

    eval_callback = EvalCallback(vec_eval_env, best_model_save_path="../../FleetRL/fleet_env/test_ev", log_path="../../FleetRL/fleet_env/test_ev",
                                 eval_freq=50, deterministic=True, render=False, verbose=1, warn=True,
                                 n_eval_episodes=5)

    n_actions = vec_train_env.action_space.shape[-1]
    param_noise = None
    noise_scale = 0.3
    seq_len = 48 * 4
    action_noise = PinkActionNoise(noise_scale, seq_len, n_actions)

    model = TD3("MlpPolicy",
                env=vec_train_env,
                verbose=0,
                train_freq=4,
                learning_rate=0.005,
                batch_size=256,
                buffer_size=1000000,
                learning_starts=10000,
                action_noise=action_noise,
                tensorboard_log=f"./tblog")

    saving_interval = 20000

    for i in range(1, 50):
        model.learn(total_timesteps=saving_interval, reset_num_timesteps=False, tb_log_name=f"TD3_{time_now}_{comment}",
                    callback=eval_callback)
        model.save(f"{trained_agents_dir}/{saving_interval * i}")

        # Don't forget to save the VecNormalize statistics when saving the agent
        log_dir = "./tmp/vec_td3/"
        model.save(log_dir + f"td3-fleet_{comment}")
        stats_path = os.path.join(log_dir, f"vec_normalize-{comment}.pkl")
        vec_train_env.save(stats_path)

    # model.learn(total_timesteps=5000, callback=eval_callback, progress_bar=True)

    # # Don't forget to save the VecNormalize statistics when saving the agent
    # log_dir = "/tmp/"
    # model.save(log_dir + "ppo_halfcheetah")
    # stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    # env.save(stats_path)
    #
    # # To demonstrate loading
    # del model, vec_env
    #
    # # Load the saved statistics
    # vec_env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
    # vec_env = VecNormalize.load(stats_path, vec_env)
    # #  do not update them at test time
    # vec_env.training = False
    # # reward normalization is not needed at test time
    # vec_env.norm_reward = False
    #
    # # Load the agent
    # model = PPO.load(log_dir + "ppo_halfcheetah", env=vec_env)