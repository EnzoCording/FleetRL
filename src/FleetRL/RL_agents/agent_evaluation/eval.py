import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from FleetRL.fleet_env.fleet_environment import FleetEnv

# %%
# wrap for multi-processing option
if __name__ == "__main__":

    # define parameters here for easier change
    n_steps = 240
    n_episodes = 1
    n_evs = 5
    n_envs = 1

    # make env for the agent
    eval_vec_env = make_vec_env(FleetEnv,
                                n_envs=n_envs,
                                vec_env_cls=SubprocVecEnv,
                                env_kwargs={
                                    "schedule_name": "ut_sched_5_eval.csv",
                                    "building_name": "load_ut.csv",
                                    "use_case": "ut",
                                    "include_building": True,
                                    "include_pv": True,
                                    "time_picker": "static",
                                    "deg_emp": False,
                                    "include_price": True,
                                    "ignore_price_reward": False,
                                    "ignore_invalid_penalty": False,
                                    "ignore_overcharging_penalty": False,
                                    "ignore_overloading_penalty": False,
                                    "episode_length": n_steps,
                                    "normalize_in_env": False,
                                    "verbose": 1,
                                    "aux": True,
                                    "log_data": True,
                                    "calculate_degradation": True
                                })
    # %%
    eval_norm_vec_env = VecNormalize(venv=eval_vec_env,
                                     norm_obs=True,
                                     norm_reward=True,
                                     training=True,
                                     clip_reward=10.0)

    dumb_vec_env = make_vec_env(FleetEnv,
                                n_envs=n_envs,
                                vec_env_cls=SubprocVecEnv,
                                env_kwargs={
                                    "schedule_name": "ut_sched_5_eval.csv",
                                    "building_name": "load_ut.csv",
                                    "use_case": "ut",
                                    "include_building": True,
                                    "include_pv": True,
                                    "time_picker": "static",
                                    "deg_emp": False,
                                    "include_price": True,
                                    "ignore_price_reward": False,
                                    "ignore_invalid_penalty": False,
                                    "ignore_overcharging_penalty": False,
                                    "ignore_overloading_penalty": False,
                                    "episode_length": n_steps,
                                    "normalize_in_env": False,
                                    "verbose": 1,
                                    "aux": True,
                                    "log_data": True,
                                    "calculate_degradation": True
                                })
    # %%
    dumb_norm_vec_env = VecNormalize(venv=dumb_vec_env,
                                     norm_obs=True,
                                     norm_reward=True,
                                     training=True,
                                     clip_reward=10.0)

    # %%
    eval_norm_vec_env.load(load_path="../trained_agents/TD3/5/UT/ut_5cars_jul5/vec_normalize-td3_new_5cars_UT_cont_on_new_env.pkl", venv=eval_norm_vec_env)
    model = TD3.load("../trained_agents/TD3/5/UT/ut_5cars_jul5/td3-fleet_td3_new_5cars_UT_cont_on_new_env.zip", eval_norm_vec_env)
    len = len(model.observation_space.low)
    model.observation_space.low = np.full(len, -np.inf)
    model.observation_space.high = np.full(len, np.inf)
    model.env = eval_norm_vec_env

    # %%
    mean_reward, _ = evaluate_policy(model, eval_norm_vec_env, n_eval_episodes=n_episodes, deterministic=True)
    print(mean_reward)

    # %%
    log_RL = model.env.env_method("get_log")[0]
    rl_start_time = model.env.env_method("get_start_time")[0]
    dumb_norm_vec_env.env_method("set_start_time", rl_start_time)
    # %%
    print("################################################################")

    episode_length = n_steps
    timesteps_per_hour = 4
    n_episodes = n_episodes
    dumb_norm_vec_env.reset()

    for i in range(episode_length * timesteps_per_hour * n_episodes):
        if dumb_norm_vec_env.env_method("is_done")[0]:
            dumb_norm_vec_env.reset()
        dumb_norm_vec_env.step([np.ones(5)])

    dumb_log = dumb_norm_vec_env.env_method("get_log")[0]

    log_RL.reset_index(drop=True, inplace=True)
    log_RL = log_RL.iloc[0:-2]
    dumb_log.reset_index(drop=True, inplace=True)
    dumb_log = dumb_log.iloc[0:-2]

    rl_cashflow = log_RL["Cashflow"].sum()
    rl_reward = log_RL["Reward"].sum()
    rl_deg = log_RL["Degradation"].sum()
    rl_overloading = log_RL["Grid overloading"].sum()
    rl_soc_violation = log_RL["SOC violation"].sum()
    rl_n_violations = log_RL[log_RL["SOC violation"] > 0]["SOC violation"].size
    rl_soh = log_RL["SOH"].iloc[-1]

    dumb_cashflow = dumb_log["Cashflow"].sum()
    dumb_reward = dumb_log["Reward"].sum()
    dumb_deg = dumb_log["Degradation"].sum()
    dumb_overloading = dumb_log["Grid overloading"].sum()
    dumb_soc_violation = dumb_log["SOC violation"].sum()
    dumb_n_violations = dumb_log[dumb_log["SOC violation"] > 0]["SOC violation"].size
    dumb_soh = dumb_log["SOH"].iloc[-1]

    print(f"RL_agents reward: {rl_reward}")
    print(f"DC reward: {dumb_reward}")
    print(f"RL_agents cashflow: {rl_cashflow}")
    print(f"DC cashflow: {dumb_cashflow}")

    total_results = pd.DataFrame()
    total_results["Category"] = ["Reward", "Cashflow", "Average degradation per EV", "Overloading", "SOC violation", "# Violations", "SOH"]

    total_results["RL_agents-based charging"] = [rl_reward,
                                          rl_cashflow,
                                          np.round(rl_deg.mean(), 5),
                                          rl_overloading,
                                          rl_soc_violation,
                                          rl_n_violations,
                                          np.round(rl_soh.mean(), 5)]

    total_results["Dumb charging"] = [dumb_reward,
                                      dumb_cashflow,
                                      np.round(dumb_deg.mean(), 5),
                                      dumb_overloading,
                                      dumb_soc_violation,
                                      dumb_n_violations,
                                      np.round(dumb_soh.mean(), 5)]

    print(total_results)


    # real charging power sent
    real_power_rl = []
    for i in range(log_RL.__len__()):
        log_RL.loc[i, "hour_id"] = (log_RL.loc[i, "Time"].hour + log_RL.loc[i, "Time"].minute / 60)
        # real_power_rl.append({"real_power": (log_RL.loc[i, "Action"]
        #                                      * log_RL.loc[i, "Observation"][2 * n_evs + 19:2 * n_evs + 19 + n_evs]
        #                                      * log_RL.loc[i, "Observation"][-4])})

    #log_RL = pd.concat((log_RL, pd.DataFrame(real_power_rl)), axis=1)

    real_power_dumb = []
    for i in range(dumb_log.__len__()):
        dumb_log.loc[i, "hour_id"] = (dumb_log.loc[i, "Time"].hour + dumb_log.loc[i, "Time"].minute / 60)
        # real_power_dumb.append({"real_power": (dumb_log.loc[i, "Action"]
        #                                        * dumb_log.loc[i, "Observation"][2 * n_evs + 19:2 * n_evs + 19 + n_evs]
        #                                        * dumb_log.loc[i, "Observation"][4 * n_evs + 19:4 * n_evs + 19 + n_evs]
        #                                        * dumb_log.loc[i, "Observation"][-4])})

    #dumb_log = pd.concat((dumb_log, pd.DataFrame(real_power_dumb)), axis=1)

    mean_per_hid_rl = log_RL.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
    mean_all_rl = []
    for i in range(mean_per_hid_rl.__len__()):
        mean_all_rl.append(np.mean(mean_per_hid_rl[i]))

    mean_per_hid_dumb = dumb_log.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
    mean_all_dumb = []
    for i in range(mean_per_hid_dumb.__len__()):
        mean_all_dumb.append(np.mean(mean_per_hid_dumb[i]))

    mean_both = pd.DataFrame()
    mean_both["RL_agents"] = np.multiply(mean_all_rl, 4)
    mean_both["Dumb charging"] = np.multiply(mean_all_dumb, 4)

    mean_both.plot()

    plt.xticks([0,8,16,24,32,40,48,56,64,72,80,88]
               ,["00:00","02:00","04:00","06:00","08:00","10:00","12:00","14:00","16:00","18:00","20:00","22:00"],
               rotation=45)

    plt.legend()
    plt.grid(alpha=0.2)

    plt.ylabel("Charging power in kW")
    max = log_RL.loc[0, "Observation"][-4]
    plt.ylim([-max * 1.2, max * 1.2])

    plt.show()
