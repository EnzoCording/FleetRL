import pandas as pd
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from FleetRL.fleet_env.fleet_environment import FleetEnv

if __name__ == "__main__":

    eval_vec_env = make_vec_env(FleetEnv,
                                n_envs=1,
                                vec_env_cls=SubprocVecEnv,
                                env_kwargs={
                                    "schedule_name": "lmd_sched_single.csv",
                                    "building_name": "load_lmd.csv",
                                    "use_case": "lmd",
                                    "include_building": True,
                                    "include_pv": True,
                                    "eval_time_picker": True,
                                    "deg_emp": False,
                                    "include_price": True,
                                    "ignore_price_reward": False,
                                    "ignore_invalid_penalty": False,
                                    "ignore_overcharging_penalty": False,
                                    "ignore_overloading_penalty": False,
                                    "episode_length": 48,
                                    "normalize_in_env": False,
                                    "verbose": 1,
                                    "aux": True,
                                    "log_data": True,
                                    "calculate_degradation": True
                                })

    eval_norm_vec_env = VecNormalize(venv=eval_vec_env,
                                     norm_obs=True,
                                     norm_reward=True,
                                     training=True,
                                     clip_reward=10.0)

    # model = TD3.load("./../trained/", env=eval_norm_vec_env)
    # model = TD3.load("./../trained/td3_aux_reworder_260000/td3_aux_reworder_260000", env=eval_norm_vec_env)
    model = PPO.load("./../trained/vec_ppo-1687255378-ppo_full_vecnorm_clip5_aux_rewardshape_order_2006_12/820000.zip", env=eval_norm_vec_env)
    # model = TD3.load("./../trained/td3_aux_lr001_2cars_run3_harsher_3rew/980000", env = eval_norm_vec_env)
    # model = TD3.load("./../trained/vec_TD3-1687291761-td3_2_cars_run2/260000", env = eval_norm_vec_env)

    # model = TD3.load("./../../../FleetRL/Output_Files/Models/TD3_aux/5cars_td3_ct/840000", env=eval_norm_vec_env)

    mean_reward, _ = evaluate_policy(model, eval_norm_vec_env, n_eval_episodes=1, deterministic=True)
    print(mean_reward)
    log_RL = model.env.env_method("get_log")[0]

    rl_start_time = model.env.env_method("get_start_time")[0]

    dumb_vec_env = make_vec_env(FleetEnv,
                                n_envs=1,
                                vec_env_cls=SubprocVecEnv,
                                env_kwargs={
                                    "schedule_name": "lmd_sched_single.csv",
                                    "building_name": "load_lmd.csv",
                                    "use_case": "lmd",
                                    "include_building": True,
                                    "include_pv": True,
                                    "eval_time_picker": True,
                                    "deg_emp": False,
                                    "include_price": True,
                                    "ignore_price_reward": False,
                                    "ignore_invalid_penalty": False,
                                    "ignore_overcharging_penalty": False,
                                    "ignore_overloading_penalty": False,
                                    "episode_length": 48,
                                    "normalize_in_env": False,
                                    "verbose": 0,
                                    "aux": True,
                                    "log_data": True,
                                    "calculate_degradation": True
                                })

    dumb_norm_vec_env = VecNormalize(venv=dumb_vec_env,
                                     norm_obs=True,
                                     norm_reward=True,
                                     training=True,
                                     clip_reward=10.0)

    dumb_norm_vec_env.env_method("set_start_time", rl_start_time)

    episode_length = 48
    timesteps_per_hour = 4
    n_episodes = 1
    dumb_norm_vec_env.reset()
    for i in range(episode_length*timesteps_per_hour*n_episodes):
        if dumb_norm_vec_env.env_method("is_done")[0]:
            dumb_norm_vec_env.reset()
        dumb_norm_vec_env.step([[1]])

    dumb_log = dumb_norm_vec_env.env_method("get_log")[0]

    rl_cashflow = log_RL["Cashflow"].sum()
    rl_reward = log_RL["Reward"].sum()
    rl_deg = log_RL["Degradation"].sum()
    rl_overloading = log_RL["Grid overloading"].sum()
    rl_soc_violation = log_RL["SOC violation"].sum()

    dumb_cashflow = dumb_log["Cashflow"].sum()
    dumb_reward = dumb_log["Reward"].sum()
    dumb_deg = dumb_log["Degradation"].sum()
    dumb_overloading = dumb_log["Grid overloading"].sum()
    dumb_soc_violation = dumb_log["SOC violation"].sum()

    print(f"RL reward: {rl_reward}")
    print(f"DC reward: {dumb_reward}")
    print(f"RL cashflow: {rl_cashflow}")
    print(f"DC cashflow: {dumb_cashflow}")

    total_results = pd.DataFrame()
    total_results["Category"] = ["Reward", "Cashflow", "Degradation", "Overloading", "SOC violation"]
    total_results["RL-based charging"] = [rl_reward, rl_cashflow, np.round(rl_deg, 5), rl_overloading, rl_soc_violation]
    total_results["Dumb charging"] = [dumb_reward, dumb_cashflow, np.round(dumb_deg, 5), dumb_overloading, dumb_soc_violation]

    print(total_results)
