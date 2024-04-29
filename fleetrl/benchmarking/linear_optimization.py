from fleetrl.fleet_env.fleet_environment import FleetEnv
from fleetrl.benchmarking.benchmark import Benchmark

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import pyomo.environ as pyo

class LinearOptimization(Benchmark):

    def __init__(self,
                 n_steps: int,
                 n_evs: int,
                 n_episodes: int = 1,
                 n_envs: int = 1,
                 time_steps_per_hour: int = 4):

        self.n_steps = n_steps
        self.n_evs = n_evs
        self.n_episodes = n_episodes
        self.n_envs = n_envs
        self.time_steps_per_hour = time_steps_per_hour

    def run_benchmark(self,
                      use_case: str,
                      env_kwargs: dict,
                      seed: int = None
                      ) -> pd.DataFrame:

        lin_vec_env = make_vec_env(FleetEnv,
                                   n_envs=self.n_envs,
                                   vec_env_cls=SubprocVecEnv,
                                   env_kwargs=env_kwargs,
                                   seed=seed)

        lin_norm_vec_env = VecNormalize(venv=lin_vec_env,
                                        norm_obs=True,
                                        norm_reward=True,
                                        training=True,
                                        clip_reward=10.0)

        env_config = env_kwargs["env_config"]

        env = FleetEnv(env_config)

        # reading the input file as a pandas DataFrame
        df: pd.DataFrame = env.db

        # adjust length of df for n_steps
        first_date = df["date"].min()
        last_date = df["date"].min() + dt.timedelta(hours=self.n_steps) - dt.timedelta(minutes=15)
        df = df[df.groupby(by="ID").date.transform(lambda x: x <= last_date)]

        # Extracting information from the df
        ev_data = [df.loc[df["ID"] == i, "There"] for i in range(self.n_evs)]
        building_data = df["load"]  # building load in kW

        # quarter hour resolution, n_steps is in hours
        length_time_load_pv = self.n_steps * 4

        price_data = np.multiply(np.add(df["DELU"], env.ev_config.fixed_markup), env.ev_config.variable_multiplier) / 1000
        tariff_data = np.multiply(df["tariff"], 1 - env.ev_config.feed_in_deduction) / 1000

        pv_data = df["pv"]  # pv power in kW
        soc_on_return = [df.loc[df["ID"] == i, "SOC_on_return"] for i in range(self.n_evs)]

        battery_capacity = env.ev_config.init_battery_cap  # EV batt size in kWh
        p_trafo = env.load_calculation.grid_connection  # Transformer rating in kW

        charging_eff = env.ev_config.charging_eff  # charging losses
        discharging_eff = env.ev_config.discharging_eff  # discharging losses

        init_soc = env.ev_config.def_soc  # init SoC

        evse_max_power = env.load_calculation.evse_max_power  # kW, max rating of the charger

        # create pyomo model
        model = pyo.ConcreteModel(name="sc_pyomo")

        model.timestep = pyo.Set(initialize=range(length_time_load_pv))
        model.time_batt = pyo.Set(initialize=range(0, length_time_load_pv + 1))
        model.ev_id = pyo.Set(initialize=range(self.n_evs))

        # model parameters
        model.building_load = pyo.Param(model.timestep,
                                        initialize={i: building_data[i] for i in range(length_time_load_pv)})
        model.pv = pyo.Param(model.timestep, initialize={i: pv_data[i] for i in range(length_time_load_pv)})
        model.ev_availability = pyo.Param(model.timestep, model.ev_id,
                                          initialize={(i, j): ev_data[j].iloc[i] for i in range(length_time_load_pv) for
                                                      j in range(self.n_evs)})
        model.soc_on_return = pyo.Param(model.timestep, model.ev_id,
                                        initialize={(i, j): soc_on_return[j].iloc[i] for i in range(length_time_load_pv)
                                                    for j in range(self.n_evs)})
        model.price = pyo.Param(model.timestep, initialize={i: price_data[i] for i in range(length_time_load_pv)})
        model.tariff = pyo.Param(model.timestep, initialize={i: tariff_data[i] for i in range(length_time_load_pv)})

        # decision variables
        # this assumes only charging, I could also make bidirectional later
        model.soc = pyo.Var(model.time_batt, model.ev_id, bounds=(0, env.ev_config.target_soc))
        model.charging_signal = pyo.Var(model.timestep, model.ev_id, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.discharging_signal = pyo.Var(model.timestep, model.ev_id, within=pyo.NonPositiveReals, bounds=(-1, 0))
        model.positive_action = pyo.Var(model.timestep, model.ev_id, within=pyo.Binary)
        model.used_pv = pyo.Var(model.timestep, model.ev_id, within=pyo.NonNegativeReals)

        def grid_limit(m, i, ev):
            return ((m.charging_signal[i, ev] + m.discharging_signal[i, ev]) * evse_max_power
                    + m.building_load[i] - m.pv[i] <= p_trafo)

        def mutual_exclusivity_charging(m, i, ev):
            return m.charging_signal[i, ev] <= m.positive_action[i, ev]

        def mutual_exclusivity_discharging(m, i, ev):
            return m.discharging_signal[i, ev] >= (m.positive_action[i, ev] - 1)

        def pv_use(m, i, ev):
            return m.used_pv[i, ev] <= m.charging_signal[i, ev] * evse_max_power

        def pv_avail(m, i, ev):
            return m.used_pv[i, ev] <= m.pv[i] / self.n_evs

        def no_charge_when_no_car(m, i, ev):
            if m.ev_availability[i, ev] == 0:
                return m.charging_signal[i, ev] == 0
            else:
                return pyo.Constraint.Feasible

        def no_discharge_when_no_car(m, i, ev):
            if m.ev_availability[i, ev] == 0:
                return m.discharging_signal[i, ev] == 0
            else:
                return pyo.Constraint.Feasible

        def soc_rules(m, i, ev):
            # last time step
            if i == length_time_load_pv - 1:
                return (m.soc[i + 1, ev]
                        == m.soc[i, ev] + (m.charging_signal[i, ev] * charging_eff + m.discharging_signal[i, ev])
                        * evse_max_power * 1 / self.time_steps_per_hour / battery_capacity)

            # new arrival
            elif (m.ev_availability[i, ev] == 0) and (m.ev_availability[i + 1, ev] == 1):
                return m.soc[i + 1, ev] == m.soc_on_return[i + 1, ev]

            # departure in next time step
            elif (m.ev_availability[i, ev] == 1) and (m.ev_availability[i + 1, ev] == 0):
                return m.soc[i, ev] == env.ev_config.target_soc

            else:
                return pyo.Constraint.Feasible

        def charging_dynamics(m, i, ev):
            # last time step
            if i == length_time_load_pv - 1:
                return (m.soc[i + 1, ev]
                        == m.soc[i, ev] + (m.charging_signal[i, ev] * charging_eff + m.discharging_signal[i, ev])
                        * evse_max_power * 1 / self.time_steps_per_hour / battery_capacity)

            # charging
            if (m.ev_availability[i, ev] == 1) and (m.ev_availability[i + 1, ev] == 1):
                return (m.soc[i + 1, ev]
                        == m.soc[i, ev] + (m.charging_signal[i, ev] * charging_eff + m.discharging_signal[i, ev])
                        * evse_max_power * 1 / self.time_steps_per_hour / battery_capacity)

            elif (m.ev_availability[i, ev] == 1) and (m.ev_availability[i + 1, ev] == 0):
                return m.soc[i + 1, ev] == 0

            elif m.ev_availability[i, ev] == 0:
                return m.soc[i, ev] == 0

            else:
                return pyo.Constraint.Feasible

        def max_charging_limit(m, i, ev):
            return m.charging_signal[i, ev] * evse_max_power <= evse_max_power * m.ev_availability[i, ev]

        def max_discharging_limit(m, i, ev):
            return m.discharging_signal[i, ev] * evse_max_power * -1 <= evse_max_power * m.ev_availability[i, ev]

        def first_soc(m, i, ev):
            return m.soc[0, ev] == init_soc

        def no_departure_abuse(m, i, ev):
            if i == length_time_load_pv - 1:
                return pyo.Constraint.Feasible
            if (m.ev_availability[i, ev] == 0) and (m.ev_availability[i - 1, ev]) == 1:
                return m.discharging_signal[i, ev] == 0
            elif (m.ev_availability[i, ev] == 1) and (m.ev_availability[i + 1, ev]) == 0:
                return m.discharging_signal[i, ev] == 0
            else:
                return pyo.Constraint.Feasible

        # constraints
        model.cs1 = pyo.Constraint(model.timestep, model.ev_id, rule=first_soc)
        model.cs2 = pyo.Constraint(model.timestep, model.ev_id, rule=grid_limit)
        model.cs3 = pyo.Constraint(model.timestep, model.ev_id, rule=max_charging_limit)
        model.cs4 = pyo.Constraint(model.timestep, model.ev_id, rule=max_discharging_limit)
        model.cs5 = pyo.Constraint(model.timestep, model.ev_id, rule=soc_rules)
        model.cs6 = pyo.Constraint(model.timestep, model.ev_id, rule=charging_dynamics)
        model.cs8 = pyo.Constraint(model.timestep, model.ev_id, rule=mutual_exclusivity_charging)
        model.cs9 = pyo.Constraint(model.timestep, model.ev_id, rule=mutual_exclusivity_discharging)
        model.cs10 = pyo.Constraint(model.timestep, model.ev_id, rule=no_charge_when_no_car)
        model.cs11 = pyo.Constraint(model.timestep, model.ev_id, rule=no_discharge_when_no_car)
        model.cs12 = pyo.Constraint(model.timestep, model.ev_id, rule=pv_use)
        model.cs13 = pyo.Constraint(model.timestep, model.ev_id, rule=pv_avail)

        timestep_set = pyo.RangeSet(0, length_time_load_pv - 1)

        def obj_fun(m):
            return (sum([((m.charging_signal[i, ev] * evse_max_power - m.used_pv[i, ev]) / self.time_steps_per_hour) *
                         m.price[i] +
                         ((m.discharging_signal[i, ev] * evse_max_power * discharging_eff) / self.time_steps_per_hour) *
                         m.tariff[i]
                         for i in m.timestep for ev in range(self.n_evs)]))

        model.obj = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
        opt = pyo.SolverFactory(
            'glpk')  # , executable="/home/enzo/Downloads/gurobi10.0.2_linux64/gurobi1002/linux64/")
        opt.options['mipgap'] = 0.005

        res = opt.solve(model, tee=True)
        print(res)

        actions = [
            np.array([model.charging_signal[i, j].value + model.discharging_signal[i, j].value for j in range(self.n_evs)])
            for i in range(length_time_load_pv)]
        actions = pd.DataFrame({"action": actions})

        actions.index = pd.date_range(start=first_date, end=last_date, freq="15T")

        actions["hid"] = actions.index.hour + actions.index.minute / 60

        len_day = 24 * 4
        action_plot = []
        for i in range(len_day):
            action_plot.append(actions.groupby("hid").mean()["action"].reset_index(drop=True)[i].mean())
        # plt.plot(action_plot)
        # plt.show()

        lin_norm_vec_env.reset()
        start_time = lin_norm_vec_env.env_method("get_start_time")[0]
        end_time = pd.to_datetime(start_time) + dt.timedelta(hours=self.n_steps)
        env_actions = actions.loc[(actions.index >= start_time) & (actions.index <= end_time), "action"].reset_index(
            drop=True)

        for i in range(self.n_steps * self.time_steps_per_hour):
            lin_norm_vec_env.step([np.multiply(np.ones(self.n_evs), env_actions[i])])

        lin_log: pd.DataFrame = lin_norm_vec_env.env_method("get_log")[0]

        lin_log.reset_index(drop=True, inplace=True)
        lin_log = lin_log.iloc[0:-2]

        return lin_log

    def plot_benchmark(self,
                       lin_log: pd.DataFrame,
                       ) -> None:

        lin_log["hour_id"] = (lin_log["Time"].dt.hour + lin_log["Time"].dt.minute / 60)

        mean_per_hid_lin = lin_log.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
        mean_all_lin = []
        for i in range(mean_per_hid_lin.__len__()):
            mean_all_lin.append(np.mean(mean_per_hid_lin[i]))

        mean = pd.DataFrame()
        mean["Distributed charging"] = np.multiply(mean_all_lin, 4)

        mean.plot()

        plt.xticks([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]
                   , ["00:00", "02:00", "04:00", "06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00",
                      "22:00"],
                   rotation=45)

        plt.legend()
        plt.grid(alpha=0.2)

        plt.ylabel("Charging power in kW")
        max = lin_log.loc[0, "Observation"][-10]
        plt.ylim([-max * 1.2, max * 1.2])

        plt.show()
