import os.path
import json

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

import seaborn as sns
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

matplotlib.rcParams.update({'font.size': 16})

from fleetrl.agent_eval.evaluation import Evaluation


class BasicEvaluation(Evaluation):

    def __init__(self, n_steps: int, n_evs: int, n_envs: int = 1, n_episodes: int = 1):
        self.n_steps = n_steps
        self.n_evs = n_evs
        self.n_envs = n_envs
        self.n_episodes = n_episodes
        self.env_kwargs = None

    @staticmethod
    def _change_param(env_kwargs: dict, key: str, val):
        if env_kwargs["env_config"].__class__ == dict:
            # env_config is a dict which we can modify directly
            env_kwargs["env_config"][key] = val
        elif env_kwargs["env_config"].__class__ == str:
            # env_config is a file path to a json file
            conf_path = env_kwargs["env_config"]
            # check that file exists and open json
            assert os.path.isfile(conf_path), "Config file path not found"
            with open(f'{conf_path}', 'r') as file:
                env_config = json.load(file)
            # write episode length parm into dict
            env_config[key] = val
            # replace path string with dict to use modified env_config from now on
            env_kwargs["env_config"] = env_config
        else:
            raise TypeError("Config parameter not specified correctly. Either dict or valid path.")
        return env_kwargs


    def evaluate_agent(self,
                       env_kwargs: dict,
                       norm_stats_path: str,
                       model_path: str,
                       seed: int = None):

        env_kwargs = self._change_param(env_kwargs=env_kwargs, key="episode_length", val=self.n_steps)

        self.env_kwargs = env_kwargs

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

    def plot_detailed_actions(self,
                              start_date: str | pd.Timestamp,
                              end_date: str | pd.Timestamp,
                              rl_log: pd.DataFrame=None,
                              uc_log: pd.DataFrame=None,
                              dist_log: pd.DataFrame=None,
                              night_log: pd.DataFrame=None):

        evse_power = self.env_kwargs["env_config"]["custom_ev_charger_power_in_kw"]

        assert any(df is not None for df in [rl_log, uc_log, dist_log, night_log]), "No log provided."

        chosen_dfs = []
        log_names = []

        if rl_log is not None:
            rl_log = rl_log[(rl_log["Time"] >= start_date) & (rl_log["Time"] <= end_date)]
            chosen_dfs.append(self._get_from_obs(rl_log))
            log_names.append("RL-based charging")

        if uc_log is not None:
            uc_log = uc_log[(uc_log["Time"] >= start_date) & (uc_log["Time"] <= end_date)]
            chosen_dfs.append(self._get_from_obs(uc_log))
            log_names.append("Uncontrolled charging")

        if dist_log is not None:
            dist_log = dist_log[(dist_log["Time"] >= start_date) & (dist_log["Time"] <= end_date)]
            chosen_dfs.append(self._get_from_obs(dist_log))
            log_names.append("Distributed charging")

        if night_log is not None:
            night_log = night_log[(night_log["Time"] >= start_date) & (night_log["Time"] <= end_date)]
            chosen_dfs.append(self._get_from_obs(log=night_log))
            log_names.append("Night charging")

        # Create a subplot with 3 rows and 1 column, without sharing the x-axis
        fig = make_subplots(rows=len(chosen_dfs)+1, cols=1, shared_xaxes=False, vertical_spacing=0.1,
                            specs=[[{'secondary_y': True}] for _ in range(len(chosen_dfs)+1)],
                            subplot_titles=(["Load, PV and Price",
                                             *[f"{log_names[i]} - Money spent: €" + str(np.round(chosen_dfs[i]["CF"].sum() * -1, 1))
                                             for i in range(len(chosen_dfs))]]),
                            column_widths=[1000], row_heights=[270 for _ in range(len(chosen_dfs)+1)])

        # Add traces for the first subplot
        fig.add_trace(go.Scatter(x=chosen_dfs[0]["Date"], y=chosen_dfs[0]["Load"], name='Building Load', legendgroup="1"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chosen_dfs[0]["Date"], y=chosen_dfs[0]["PV"], name='PV', legendgroup="1"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chosen_dfs[0]["Date"], y=chosen_dfs[0]["Price"], name='Price', legendgroup="1"), row=1, col=1, secondary_y=True)

        # # Add traces for the second subplot (you can change these as needed)
        # fig.add_trace(go.Scatter(x=df["Date"], y=df_real["Action"], name='Charging power', legendgroup="2"), row=2, col=1)
        # fig.add_trace(go.Scatter(x=df["Date"], y=df_real["SOC"], name='SOC', legendgroup="2"), row=2, col=1, secondary_y=True)

        for i in range(len(chosen_dfs)):

            # Add traces for the second subplot (you can change these as needed)
            fig.add_trace(go.Scatter(x=chosen_dfs[i]["Date"],
                                     y=chosen_dfs[i]["Action"],
                                     name='Charging power',
                                     legendgroup=f"{i+2}"),
                          row=i+2, col=1)

            fig.add_trace(go.Scatter(x=chosen_dfs[i]["Date"],
                                     y=chosen_dfs[i]["SOC"],
                                     name='SOC',
                                     legendgroup=f"{i+2}"),
                          row=i+2, col=1, secondary_y=True)

        fig.update_xaxes(row=1, range=[start_date, end_date], matches="x")
        fig.update_xaxes(row=2, range=[start_date, end_date], matches="x")

        for i in range(len(chosen_dfs)+1):
            fig.update_xaxes(row=i + 1, col=1, matches='x')

        for i in range(1, len(chosen_dfs)+1):
            fig.update_yaxes(
                tickvals=[-evse_power, 0, evse_power],  # Values at which ticks on this axis appear
                ticktext=[f'-{evse_power}', '0', f'{evse_power}'],  # Text that appears at the ticks
                row=i + 1, col=1,  # Row and column of the subplot to update (adjust as needed)
                secondary_y=False  # Set to True if updating the secondary y-axis
            )

        for i in range(1, len(chosen_dfs)+1):
            fig.update_yaxes(
                tickvals=[0, 0.8],  # Values at which ticks on this axis appear
                ticktext=['0', '0.8'],  # Text that appears at the ticks
                row=i + 1, col=1,  # Row and column of the subplot to update (adjust as needed)
                secondary_y=True  # Set to True if updating the secondary y-axis
            )

        min_price = chosen_dfs[0]["Price"].min()
        max_price = chosen_dfs[0]["Price"].max()

        for i in range(1):
            fig.update_yaxes(
                tickvals = [min_price, max_price],  # Values at which ticks on this axis appear
                ticktext = [f'{np.round(min_price, 0)}', f'{np.round(max_price, 0)}'],  # Text that appears at the ticks
                row = i + 1, col=1,  # Row and column of the subplot to update (adjust as needed)
                secondary_y = True  # Set to True if updating the secondary y-axis
            )

        fig.update_yaxes(row=1, col=1, range=[min_price-2, max_price+2], secondary_y=True)
        for i in range(1, len(chosen_dfs)+1):
            fig.update_yaxes(row=i+1, col=1, range=[-evse_power*1.1, evse_power*1.1], secondary_y=False)

        # Labels for primary y-axes
        primary_labels = ["kW" for _ in range(len(chosen_dfs)+1)]

        # Labels for secondary y-axes
        secondary_labels = ["ct/kWh", *["SOC" for _ in range(len(chosen_dfs))]]

        # Update primary y-axes labels
        for i, label in enumerate(primary_labels):
            fig.update_yaxes(title_text=label, row=i + 1, col=1, secondary_y=False)

        # Update secondary y-axes labels
        for i, label in enumerate(secondary_labels):
            fig.update_yaxes(title_text=label, row=i + 1, col=1, secondary_y=True)

        fig.update_layout(
            # width=1050,
            # height=1400,
            margin=dict(l=35, r=45, t=25, b=25),
            font=dict(size=16)
        )

        fig.update_layout(
            legend_tracegroupgap=275,

        )

        # Update the x-axis to show the date without the year
        fig.update_xaxes(
            tickformat="%m-%d",  # Display only month and day
        )

        # Other plot updates
        # ...
        fig.update_xaxes(
            tickformat="%m/%d %H:%M"  # Display abbreviated month name, day, hours, and minutes
        )

        # return the plot
        return fig

    def _get_from_obs(self, log: dict):

        obs = log["Observation"]
        act = log["Charging energy"]
        cf = log["Cashflow"]
        env_config = self.env_kwargs["env_config"]

        bl_pv_lookahead = env_config["bl_pv_lookahead"]
        pr_lookahead = env_config["price_lookahead"]

        length = len(log)

        # Check observer class to see how observation list is built up

        date = log["Time"]
        first = 0  # first entry has index 0
        last = self.n_evs - 1  # soc for each car
        if self.n_evs > 1:
            soc = [obs[i][first:last].mean() for i in range(length)]
        else:
            soc = [obs[i][first] for i in range(length)]

        first = self.n_evs
        last = self.n_evs * 2 - 1  # hours left at charger for each car
        if self.n_evs > 1:
            hours_left = [obs[i][first:last].mean() for i in range(length)]
        else:
            hours_left = [obs[i][first] for i in range(length)]

        first = self.n_evs * 2
        last = self.n_evs * 2 + pr_lookahead  # price lookahead gives price in hour, hour+1, etc.
        price = [obs[i][first] for i in range(length)]

        first = self.n_evs * 2 + pr_lookahead + 1
        last = self.n_evs * 2 + pr_lookahead * 2 + 1  # tariff paid when discharging, with lookahead
        tariff = [obs[i][first] for i in range(length)]

        first = self.n_evs * 2 + pr_lookahead * 2 + 2
        last = self.n_evs * 2 + pr_lookahead * 2 + 2 + bl_pv_lookahead  # building load lookahead
        building_load = [obs[i][first] for i in range(length)]

        first = self.n_evs * 2 + pr_lookahead * 2 + 2 + bl_pv_lookahead + 1
        last = self.n_evs * 2 + pr_lookahead * 2 + bl_pv_lookahead * 2 + 1  # pv has same lookahead as building
        pv = [obs[i][first] for i in range(length)]

        free_cap = [obs[i][-8] / obs[i][-9] for i in range(length)]  # free grid capacity in MW / total grid capacity

        time_steps_per_hour = env_config["time_steps_per_hour"]

        # act is charging energy in kWh, we want to display the currently drawn power in kW
        first = 0  # first entry has index 0
        last = self.n_evs - 1  # soc for each car
        if self.n_evs > 1:
            action = [act[i][first:last].sum() * time_steps_per_hour for i in range(length)]  # Going from kWh to kW
        else:
            action = [act[i][first] * time_steps_per_hour for i in range(length)]  # Going from kWh to kW

        df = pd.DataFrame({
            'Date': date,
            'SOC': soc,
            'Load': building_load,
            'PV': pv,
            'Price': price,
            'Action': action,
            'Free cap': free_cap,
            'CF': cf
        })

        return df
