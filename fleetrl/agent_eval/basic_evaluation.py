from fleetrl.fleet_env.fleet_environment import FleetEnv
from fleetrl.benchmarking.benchmark import Benchmark

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates

from fleetrl.agent_eval.evaluation import Evaluation


class BasicEvaluation(Evaluation):

    def __init__(self, n_steps: int, n_evs: int, n_envs: int = 1, n_episodes: int = 1):
        self.n_steps = n_steps
        self.n_evs = n_evs
        self.n_envs = n_envs
        self.n_episodes = n_episodes

    def evaluate_agent(self,
                       env_kwargs: dict,
                       norm_stats_path: str,
                       model_path: str,
                       seed: int = None):

        env_kwargs["env_config"]["episode_length"] = self.n_steps

        eval_vec_env = make_vec_env(FleetEnv,
                                    n_envs=self.n_envs,
                                    vec_env_cls=SubprocVecEnv,
                                    seed=seed,
                                    env_kwargs=env_kwargs)

        eval_norm_vec_env = VecNormalize(venv=eval_vec_env,
                                         norm_obs=True,
                                         norm_reward=True,
                                         training=True,
                                         clip_reward=10.0)

        eval_norm_vec_env.load(load_path=norm_stats_path, venv=eval_norm_vec_env)
        model = PPO.load(model_path, env=eval_norm_vec_env,
                         custom_objects={"observation_space": eval_norm_vec_env.observation_space,
                                         "action_space": eval_norm_vec_env.action_space})

        mean_reward, _ = evaluate_policy(model, eval_norm_vec_env, n_eval_episodes=self.n_episodes, deterministic=True)
        print(mean_reward)

        log_RL = model.env.env_method("get_log")[0]
        log_RL.reset_index(drop=True, inplace=True)
        log_RL = log_RL.iloc[0:-2]

        return log_RL

    def compare(self, rl_log: pd.DataFrame, benchmark_log: pd.DataFrame):

        rl_cashflow = rl_log["Cashflow"].sum()
        rl_reward = rl_log["Reward"].sum()
        rl_deg = rl_log["Degradation"].sum()
        rl_overloading = rl_log["Grid overloading"].sum()
        rl_soc_violation = rl_log["SOC violation"].sum()
        rl_n_violations = rl_log[rl_log["SOC violation"] > 0]["SOC violation"].size
        rl_soh = rl_log["SOH"].iloc[-1]

        benchmark_cashflow = benchmark_log["Cashflow"].sum()
        benchmark_reward = benchmark_log["Reward"].sum()
        benchmark_deg = benchmark_log["Degradation"].sum()
        benchmark_overloading = benchmark_log["Grid overloading"].sum()
        benchmark_soc_violation = benchmark_log["SOC violation"].sum()
        benchmark_n_violations = benchmark_log[benchmark_log["SOC violation"] > 0]["SOC violation"].size
        benchmark_soh = benchmark_log["SOH"].iloc[-1]

        print(f"RL reward: {rl_reward}")
        print(f"DC reward: {benchmark_reward}")
        print(f"RL cashflow: {rl_cashflow}")
        print(f"DC cashflow: {benchmark_cashflow}")

        total_results = pd.DataFrame()
        total_results["Category"] = ["Reward", "Cashflow", "Average degradation per EV", "Overloading", "SOC violation",
                                     "# Violations", "SOH"]

        total_results["RL-based charging"] = [rl_reward,
                                              rl_cashflow,
                                              np.round(np.mean(rl_deg), 5),
                                              rl_overloading,
                                              rl_soc_violation,
                                              rl_n_violations,
                                              np.round(np.mean(rl_soh), 5)]

        total_results["benchmark charging"] = [benchmark_reward,
                                               benchmark_cashflow,
                                               np.round(np.mean(benchmark_deg), 5),
                                               benchmark_overloading,
                                               benchmark_soc_violation,
                                               benchmark_n_violations,
                                               np.round(np.mean(benchmark_soh), 5)]

        print(total_results)

        rl_log["hour_id"] = (rl_log["Time"].dt.hour + rl_log["Time"].dt.minute / 60)

        benchmark_log["hour_id"] = (benchmark_log["Time"].dt.hour + benchmark_log["Time"].dt.minute / 60)

        mean_per_hid_rl = rl_log.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
        mean_all_rl = []
        for i in range(mean_per_hid_rl.__len__()):
            mean_all_rl.append(np.mean(mean_per_hid_rl[i]))

        mean_per_hid_benchmark = benchmark_log.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
        mean_all_benchmark = []
        for i in range(mean_per_hid_benchmark.__len__()):
            mean_all_benchmark.append(np.mean(mean_per_hid_benchmark[i]))

        mean_both = pd.DataFrame()
        mean_both["RL"] = np.multiply(mean_all_rl, 4)
        mean_both["benchmark charging"] = np.multiply(mean_all_benchmark, 4)

        mean_both.plot()

        plt.xticks([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]
                   , ["00:00", "02:00", "04:00", "06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00",
                      "22:00"],
                   rotation=45)

        plt.legend()
        plt.grid(alpha=0.2)

        plt.ylabel("Charging power in kW")
        max = rl_log.loc[0, "Observation"][-10]
        plt.ylim([-max * 1.2, max * 1.2])

        plt.show()

    def plot_soh(self, rl_log, benchmark_log):

        # Create a date range from Jan to Dec
        # Create a date range from Jan to Dec with a 15-minute resolution
        date_range = pd.date_range(start=rl_log["Time"].iloc[0], end=rl_log["Time"].iloc[-1], freq='15min')

        # Create a figure
        fig, ax = plt.subplots()

        # Rescale the index of the dataframes to match the date range
        rescaled_rl_log = rl_log.copy()
        rescaled_rl_log.index = date_range[:len(rl_log)]
        rescaled_benchmark_log = benchmark_log.copy()
        rescaled_benchmark_log.index = date_range[:len(benchmark_log)]

        # Plot the data
        ax.plot(rescaled_benchmark_log.index, rescaled_benchmark_log['SOH'].apply(lambda x: x[0]), label='Dumb', color='red')
        ax.plot(rescaled_rl_log.index, rescaled_rl_log['SOH'].apply(lambda x: x[0]), label='RL', color='blue')

        # Set the title and labels
        ax.set_title('State of Health Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('State of Health')
        ax.legend()

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        ax.set_xticklabels(ax.get_xticklabels()[0:12])[0:12]
        ax.set_xticks(ax.get_xticks()[0:12])[0:12]
        # Show the plot
        plt.grid(alpha=0.2)

        plt.show()

    def plot_violations(self, rl_log, benchmark_log):
        
        fig, axs = plt.subplots(1, 1, figsize=(3, 3))

        # Plot RL data
        rl_log.loc[rl_log["SOC violation"] > 0, "SOC violation"].sort_values(ascending=False).reset_index(
            drop=True).plot.bar(color="blue", ax=axs)
        axs.set_title("RL-based charging")
        axs.set_xticks([0, 25, 50, 75, 100])
        axs.set_ylim([0, 0.4])
        axs.grid(alpha=0.2)
        axs.set_xlabel("Violations")
        axs.set_ylabel("Missing SOC per violation")

        plt.tight_layout()
        plt.show()

    def plot_action_dist(self, rl_log, benchmark_log):

        if rl_log['Action'][0].__class__ == np.ndarray:
            rl_log['Action'] = rl_log['Action'].apply(lambda x: x[0])
            benchmark_log['Action'] = benchmark_log['Action'].apply(lambda x: x[0])

        # Create a figure with two subplots side by side
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))

        # Plot the distribution of actions for the RL-based strategy on the first subplot
        axs[0].hist(rl_log['Action'], bins=50, color='blue', edgecolor='black')
        axs[0].set_title('Distribution of Actions for RL-based Charging')
        axs[0].set_xlabel('Action')
        axs[0].set_ylabel('Frequency')

        # Plot the distribution of actions for the benchmark strategy on the second subplot
        axs[1].hist(benchmark_log['Action'], bins=50, color='red', edgecolor='black')
        axs[1].set_title('Distribution of actions for benchmark Charging')
        axs[1].set_xlabel('Action')
        axs[1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()
