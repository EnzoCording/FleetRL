import random
import matplotlib.pyplot as plt
import rainflow as rf
import pandas as pd
import numpy as np
from FleetRL.fleet_env.fleet_environment import FleetEnv
# from FleetRL.utils.prices import load_prices
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

env_args = {"include_pv": True,
            "schedule_name": "lmd_sched_single.csv",
            "use_case": "lmd",
            "include_building": True,
            "include_price": True,
            "normalize_in_env": False,
            "aux": True,
            "calculate_degradation": True,
            "log_data": True,
            "episode_length": 24,
            "time_picker": "static",
            }


env = FleetEnv(log_data=True,
               include_pv=True,
               schedule_name="lmd_sched_single.csv",
               use_case="lmd",
               building_name="load_ct.csv",
               price_name="spot_2021_new.csv",
               tariff_name="fixed_feed_in.csv",
               normalize_in_env=False,
               aux=True, 
               calculate_degradation=True,
               time_picker="static",
               episode_length=48,
               include_building=True,
               spot_markup=10,
               spot_mul=1.5,
               feed_in_ded=0.25)

env.reset();
#
# for i in range (90): env.step([0,0,0,0,0]);

# for i in range(96):
#     env.step([-1,-1,-1,-1,-1])
#

'''
env = make_vec_env(FleetEnv, env_kwargs=env_args)

norm_env = VecNormalize(env, norm_reward=True, norm_obs=True, training=True)

model = TD3(policy="MlpPolicy", env=norm_env, verbose=1, tensorboard_log="./tb_log")
model.learn(total_timesteps=20000, tb_log_name="./20k_test")
model.save("20k_test")

eval_norm_vec_env = make_vec_env(FleetEnv, env_kwargs=env_args)
dumb_norm_vec_env = make_vec_env(FleetEnv, env_kwargs=env_args)

model = TD3.load("./20k_test", env = eval_norm_vec_env)
mean_reward, _ = evaluate_policy(model, eval_norm_vec_env, n_eval_episodes=5, deterministic=True)

log_RL = model.env.env_method("get_log")[0]
rl_start_time = model.env.env_method("get_start_time")[0]
dumb_norm_vec_env.env_method("set_start_time", rl_start_time)
# %%
print("################################################################")

episode_length = 24
timesteps_per_hour = 4
n_episodes = 1
n_evs = 1
dumb_norm_vec_env.reset()

for i in range(episode_length * timesteps_per_hour * n_episodes):
    if dumb_norm_vec_env.env_method("is_done")[0]:
        dumb_norm_vec_env.reset()
    dumb_norm_vec_env.step([np.ones(n_evs)])

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
'''
