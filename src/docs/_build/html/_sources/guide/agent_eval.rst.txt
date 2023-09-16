.. _agent_eval:

Agent Evaluation
============

Agent evaluation requires a trained RL agent, in form of a .zip artifact generated from SB3.
The evaluation features a comparison with uncontrolled charging to allow for a first basic
comparison. It is optional and can be toggled off to save compute time.

**Import requirements**

.. code-block:: python

    import datetime

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3 import TD3, PPO
    from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from FleetRL.fleet_env.fleet_environment import FleetEnv

**Multi-processing wrapper**

The evaluation uses vectorized environments that allow for parallelization. If this script is
run in a Jupyter notebook, no changes need to be made. If, however, the script is run as a .py
file, it needs to be wrapped as follows:

.. code-block:: python

    if __name__ == "__main__":
        # code here

**Defining fundamental parameters**

By default, ``n_steps`` is set to 8600. This means that the evaluation episode is set to 8600
hours. The trained agent is therefore tested on one year of unseen schedule data. A separately
generated schedule is used that the agent did not see during training.

.. code-block:: python

    # define parameters here for easier change
    n_steps = 8600
    n_episodes = 1
    n_evs = 1
    n_envs = 1
    file_name_comment = "comment"  # added to log pickle file names
**Environment creation**

The testing environment is created. The parameters are the same as for training - only
the schedule differs: ``lmd_sched_single_eval.csv``. The same normalization and vectorization is performed.
Generally, agents are cross-compatible with environments of same dimension and boundaries. A 1-car agent
trained on the environment for caretakers can thus be used on the last-mile delivery environment, if
the ``gym.Spaces`` bounds are the same. If normalization is conducted in FleetRL, the bounds are [0,1]. If
no normalization is conducted, the bounds are [-inf, inf]. This ensures maximum cross-compatibility.
Similarly, the environment for the uncontrolled charging agent is created (``dumb_vec_env, dumb_norm_vec_env``).

.. code-block:: python

    # make env for the agent
    eval_vec_env = make_vec_env(FleetEnv,
                                n_envs=n_envs,
                                vec_env_cls=SubprocVecEnv,
                                seed=0,
                                env_kwargs={
                                    "schedule_name": "lmd_sched_single_eval.csv",  # separate testing schedule
                                    "building_name": "load_lmd.csv",
                                    "price_name": "spot_2021_new.csv",
                                    "tariff_name": "spot_2021_new_tariff.csv",
                                    "use_case": "lmd",
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
                                    "verbose": 0,
                                    "aux": True,
                                    "log_data": True,
                                    "calculate_degradation": True,
                                    "spot_markup": 0,
                                    "spot_mul": 1,
                                    "feed_in_ded": 0
                                })

    eval_norm_vec_env = VecNormalize(venv=eval_vec_env,
                                     norm_obs=True,
                                     norm_reward=True,
                                     training=True,
                                     clip_reward=10.0)

    dumb_vec_env = make_vec_env(FleetEnv,
                                n_envs=n_envs,
                                vec_env_cls=SubprocVecEnv,
                                seed=0,
                                env_kwargs={
                                    "schedule_name": "lmd_sched_single_eval.csv",
                                    "building_name": "load_lmd.csv",
                                    "price_name": "spot_2021_new.csv",
                                    "tariff_name": "spot_2021_new_tariff.csv",
                                    "use_case": "lmd",
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
                                    "verbose": 0,
                                    "aux": True,
                                    "log_data": True,
                                    "calculate_degradation": True,
                                    "spot_markup": 0,
                                    "spot_mul": 1,
                                    "feed_in_ded": 0
                                })

    dumb_norm_vec_env = VecNormalize(venv=dumb_vec_env,
                                     norm_obs=True,
                                     norm_reward=True,
                                     training=True,
                                     clip_reward=10.0)

**Loading models**

The normalization metrics can be loaded via ``VecEnv.load(load_path, venv)``. This is optional.
The RL agent is loaded. The path to the .zip artifact and the environment must be specified.
Optionally, a custom_objects parameter can be parsed to make sure that observation and action space
are correctly configured.

.. code-block:: python

    eval_norm_vec_env.load(load_path="./tmp/vec_PPO/vec_normalize-LMD_2021_arbitrage_PPO_mul3.pkl", venv=eval_norm_vec_env)
    model = PPO.load("./tmp/vec_PPO/PPO-fleet_LMD_2021_arbitrage_PPO_mul3.zip", env = eval_norm_vec_env,
                    custom_objects={"observation_space": eval_norm_vec_env.observation_space,
                                   "action_space": eval_norm_vec_env.action_space})

**RL agent evaluation**

Agents are evaluated via ``evaluate_policy``. The model, the environment, the number of episodes and the
deterministic flag are parsed. ``deterministic=True`` ensures that several evaluations of the same
agents yield the same results - ensuring reproducibility. Random fluctuations due to random number generators
or statistical distributions are eliminated.

.. code-block:: python

    mean_reward, _ = evaluate_policy(model, eval_norm_vec_env, n_eval_episodes=n_episodes, deterministic=True)
    print(mean_reward)

Once ``evaluate_policy`` concluded, the environment stepped through 8600 hours. Meanwhile,
FleetRL logged every important metric, allowing for post-processing and thorough analyses.
These can be accessed via ``env_method("get_log")[0]``, as shown below.

.. code-block:: python

    log_RL = model.env.env_method("get_log")[0]

**Uncontrolled charging agent**

The start time of the evaluation environment is extracted and set as start time for the
uncontrolled charging environment. The environment is then stepped through for the same amount
of time steps and the log is extracted.

.. code-block:: python

    # start date extraction and setting the same date to the uncontrolled charging env
    rl_start_time = model.env.env_method("get_start_time")[0]
    dumb_norm_vec_env.env_method("set_start_time", rl_start_time)

    print("################################################################")

    episode_length = n_steps
    timesteps_per_hour = 4
    n_episodes = n_episodes
    dumb_norm_vec_env.reset()

    # uncontrolled charging agent: action of "1" is sent for each time step -> charging immediately upon arrival
    for i in range(episode_length * timesteps_per_hour * n_episodes):
        if dumb_norm_vec_env.env_method("is_done")[0]:
            dumb_norm_vec_env.reset()
        dumb_norm_vec_env.step([np.ones(n_evs)])

    # log extraction from the vec_env
    dumb_log = dumb_norm_vec_env.env_method("get_log")[0]

**Post-processing**

Once both agents ran in the environments and the logs have been extracted, they can be used to
extract useful information on charging expenses, state of health, violations, rewards, etc.

.. code-block:: python

    # index reset and the last row of the dataframe is removed
    log_RL.reset_index(drop=True, inplace=True)
    log_RL = log_RL.iloc[0:-2]
    dumb_log.reset_index(drop=True, inplace=True)
    dumb_log = dumb_log.iloc[0:-2]

    # computing key performance metrics
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

    print(f"RL reward: {rl_reward}")
    print(f"DC reward: {dumb_reward}")
    print(f"RL cashflow: {rl_cashflow}")
    print(f"DC cashflow: {dumb_cashflow}")

    total_results = pd.DataFrame()
    total_results["Category"] = ["Reward", "Cashflow", "Average degradation per EV", "Overloading", "SOC violation", "# Violations", "SOH"]

    total_results["RL-based charging"] = [rl_reward,
                                          rl_cashflow,
                                          np.round(np.mean(rl_deg), 5),
                                          rl_overloading,
                                          rl_soc_violation,
                                          rl_n_violations,
                                          np.round(np.mean(rl_soh), 5)]

    total_results["Dumb charging"] = [dumb_reward,
                                      dumb_cashflow,
                                      np.round(np.mean(dumb_deg), 5),
                                      dumb_overloading,
                                      dumb_soc_violation,
                                      dumb_n_violations,
                                      np.round(np.mean(dumb_soh), 5)]

    print(total_results)


**Plotting**

As an example, the charging strategies of the RL agent and the uncontrolled charging strategy are plotted - the mean
of each quarter hour is plotted, indicating when charging signals are sent to the battery.

.. code-block:: python

    # real charging power sent to the battery
    real_power_rl = []
    for i in range(log_RL.__len__()):
        log_RL.loc[i, "hour_id"] = (log_RL.loc[i, "Time"].hour + log_RL.loc[i, "Time"].minute / 60)

    real_power_dumb = []
    for i in range(dumb_log.__len__()):
        dumb_log.loc[i, "hour_id"] = (dumb_log.loc[i, "Time"].hour + dumb_log.loc[i, "Time"].minute / 60)

    # computing the average for each quarter hour over the entire year
    mean_per_hid_rl = log_RL.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
    mean_all_rl = []
    for i in range(mean_per_hid_rl.__len__()):
        mean_all_rl.append(np.mean(mean_per_hid_rl[i]))

    mean_per_hid_dumb = dumb_log.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
    mean_all_dumb = []
    for i in range(mean_per_hid_dumb.__len__()):
        mean_all_dumb.append(np.mean(mean_per_hid_dumb[i]))

    # multiplied by the factor of 4 to go from kWh to kW (15 min time resolution)
    mean_both = pd.DataFrame()
    mean_both["RL"] = np.multiply(mean_all_rl, 4)
    mean_both["Dumb charging"] = np.multiply(mean_all_dumb, 4)

    # plotting
    mean_both.plot()
    plt.xticks([0,8,16,24,32,40,48,56,64,72,80,88],
               ["00:00","02:00","04:00","06:00","08:00","10:00","12:00","14:00","16:00","18:00","20:00","22:00"],
               rotation=45)

    plt.legend()
    plt.grid(alpha=0.2)
    plt.ylabel("Charging power in kW")
    max = log_RL.loc[0, "Observation"][-10]
    plt.ylim([-max * 1.2, max * 1.2])
    plt.show()

**Saving the logs for future use**

The logs can be saved as pickle files, so the same analytics and other visualizations can be performed on another machine,
or at a later point in time.

.. code-block:: python

    dumb_log.to_pickle(f"dumb_log_{file_name_comment}")
    log_rl.to_pickle(f"rl_log_{file_name_comment}")
