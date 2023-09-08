
import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Literal

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from FleetRL.fleet_env.fleet_environment import FleetEnv

import pandas as pd
import pyomo.environ as pyo
#import gurobipy

if __name__ == "__main__":

    # define parameters here for easier change
    n_steps = 8600
    n_episodes = 1
    n_evs = 5
    n_envs = 1
    time_steps_per_hour = 4
    use_case: str = "lmd"  # for file name
    scenario: Literal["arb", "real"] = "real"

    # environment arguments
    env_kwargs = {"schedule_name": "5_lmd_eval.csv",
                  "building_name": "load_lmd.csv",
                  "use_case": "lmd",
                  "include_building": True,
                  "include_pv": True,
                  "time_picker": "static",
                  "deg_emp": False,
                  "include_price": True,
                  "ignore_price_reward": False,
                  "ignore_invalid_penalty": True,
                  "ignore_overcharging_penalty": True,
                  "ignore_overloading_penalty": False,
                  "episode_length": n_steps,
                  "normalize_in_env": False,
                  "verbose": 0,
                  "aux": True,
                  "log_data": True,
                  "calculate_degradation": True
                  }

    if scenario == "real":
        env_kwargs["spot_markup"] = 10
        env_kwargs["spot_mul"] = 1.5
        env_kwargs["feed_in_ded"] = 0.25
        env_kwargs["price_name"] = "spot_2021_new_tariff.csv"
        env_kwargs["tariff_name"] = "fixed_feed_in.csv"
    elif scenario == "arb":
        env_kwargs["spot_markup"] = 0
        env_kwargs["spot_mul"] = 1
        env_kwargs["feed_in_ded"] = 0
        env_kwargs["price_name"] = "spot_2021_new.csv"
        env_kwargs["tariff_name"] = "spot_2021_new.csv"

    lin_vec_env = make_vec_env(FleetEnv,
                               n_envs=n_envs,
                               vec_env_cls=SubprocVecEnv,
                               env_kwargs=env_kwargs)

    lin_norm_vec_env = VecNormalize(venv=lin_vec_env,
                                    norm_obs=True,
                                    norm_reward=True,
                                    training=True,
                                    clip_reward=10.0)

    env = FleetEnv(use_case=use_case,
                   schedule_name=env_kwargs["schedule_name"],
                   tariff_name=env_kwargs["tariff_name"],
                   price_name=env_kwargs["price_name"],
                   episode_length=n_steps,
                   time_picker_name=env_kwargs["time_picker"],
                   building_name=env_kwargs["building_name"],
                   spot_markup=env_kwargs["spot_markup"],
                   spot_mul=env_kwargs["spot_mul"],
                   feed_in_ded=env_kwargs["feed_in_ded"])

    # reading the input file as a pandas DataFrame
    df: pd.DataFrame = env.db

    # Extracting information from the df
    ev_data = [df.loc[df["ID"]==i, "There"] for i in range(n_evs)]
    building_data = df["load"]  # building load in kW

    length_time_load_pv = 8760 * 4

    price_data = np.multiply(np.add(df["DELU"], env.ev_conf.fixed_markup), env.ev_conf.variable_multiplier) / 1000
    tariff_data = np.multiply(df["tariff"], 1-env.ev_conf.feed_in_deduction) / 1000

    pv_data = df["pv"]  # pv power in kW
    soc_on_return = [df.loc[df["ID"]==i, "SOC_on_return"] for i in range(n_evs)]

    battery_capacity = env.ev_conf.init_battery_cap  # EV batt size in kWh
    p_trafo = env.load_calculation.grid_connection  # Transformer rating in kW

    charging_eff = env.ev_conf.charging_eff  # charging losses
    discharging_eff = env.ev_conf.discharging_eff  # discharging losses

    init_soc = env.ev_conf.def_soc  # init SoC

    evse_max_power = env.load_calculation.evse_max_power  # kW, max rating of the charger

    # create pyomo model
    model = pyo.ConcreteModel(name="sc_pyomo")

    model.timestep = pyo.Set(initialize=range(length_time_load_pv))
    model.time_batt = pyo.Set(initialize=range(0, length_time_load_pv+1))
    model.ev_id = pyo.Set(initialize=range(n_evs))

    # model parameters
    model.building_load = pyo.Param(model.timestep, initialize={i: building_data[i] for i in range(length_time_load_pv)})
    model.pv = pyo.Param(model.timestep, initialize={i: pv_data[i] for i in range(length_time_load_pv)})
    model.ev_availability = pyo.Param(model.timestep, model.ev_id, initialize={(i, j): ev_data[j].iloc[i] for i in range(length_time_load_pv) for j in range(n_evs)})
    model.soc_on_return = pyo.Param(model.timestep, model.ev_id, initialize={(i, j): soc_on_return[j].iloc[i] for i in range(length_time_load_pv) for j in range(n_evs)})
    model.price = pyo.Param(model.timestep, initialize={i: price_data[i] for i in range(length_time_load_pv)})
    model.tariff = pyo.Param(model.timestep, initialize={i: tariff_data[i] for i in range(length_time_load_pv)})

    # decision variables
    # this assumes only charging, I could also make bidirectional later
    model.soc = pyo.Var(model.time_batt, model.ev_id, bounds=(0, env.ev_conf.target_soc))
    model.charging_signal = pyo.Var(model.timestep, model.ev_id, within=pyo.NonNegativeReals, bounds=(0,1))
    model.discharging_signal = pyo.Var(model.timestep, model.ev_id, within=pyo.NonPositiveReals, bounds=(-1,0))
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
        return m.used_pv[i, ev] <= m.pv[i] / n_evs

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
        #last time step
        if i == length_time_load_pv-1:
            return (m.soc[i+1, ev]
                    == m.soc[i, ev] + (m.charging_signal[i, ev]*charging_eff + m.discharging_signal[i, ev])
                    * evse_max_power * 1 / time_steps_per_hour / battery_capacity)

        # new arrival
        elif (m.ev_availability[i, ev] == 0) and (m.ev_availability[i+1, ev] == 1):
            return m.soc[i+1, ev] == m.soc_on_return[i+1, ev]

        # departure in next time step
        elif (m.ev_availability[i, ev] == 1) and (m.ev_availability[i+1, ev] == 0):
            return m.soc[i, ev] == env.ev_conf.target_soc

        else:
            return pyo.Constraint.Feasible

    def charging_dynamics(m, i, ev):
        #last time step
        if i == length_time_load_pv-1:
            return (m.soc[i+1, ev]
                    == m.soc[i, ev] + (m.charging_signal[i, ev]*charging_eff + m.discharging_signal[i, ev])
                    * evse_max_power * 1 / time_steps_per_hour / battery_capacity)

        # charging
        if (m.ev_availability[i, ev] == 1) and (m.ev_availability[i+1, ev] == 1):
            return (m.soc[i+1, ev]
                    == m.soc[i, ev] + (m.charging_signal[i, ev]*charging_eff + m.discharging_signal[i, ev])
                    * evse_max_power * 1 / time_steps_per_hour / battery_capacity)

        elif (m.ev_availability[i, ev] == 1) and (m.ev_availability[i+1, ev] == 0):
            return m.soc[i+1, ev] == 0

        elif m.ev_availability[i, ev] == 0:
            return m.soc[i, ev] == 0

        else:
            return pyo.Constraint.Feasible

    def max_charging_limit(m, i, ev):
        return m.charging_signal[i, ev]*evse_max_power <= evse_max_power * m.ev_availability[i, ev]

    def max_discharging_limit(m, i, ev):
        return m.discharging_signal[i, ev]*evse_max_power*-1 <= evse_max_power * m.ev_availability[i, ev]

    def first_soc(m, i, ev):
        return m.soc[0, ev] == init_soc

    def no_departure_abuse(m, i, ev):
        if i == length_time_load_pv - 1:
            return pyo.Constraint.Feasible
        if (m.ev_availability[i, ev] == 0) and (m.ev_availability[i-1, ev]) == 1:
            return m.discharging_signal[i, ev] == 0
        elif (m.ev_availability[i, ev] == 1) and (m.ev_availability[i+1, ev]) == 0:
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

    timestep_set = pyo.RangeSet(0, length_time_load_pv-1)

    def obj_fun(m):
        return (sum([((m.charging_signal[i, ev] * evse_max_power - m.used_pv[i, ev]) / time_steps_per_hour) * m.price[i] +
                     ((m.discharging_signal[i, ev] * evse_max_power * discharging_eff) / time_steps_per_hour) * m.tariff[i]
                     for i in m.timestep for ev in range(n_evs)]))

    model.obj = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
    opt = pyo.SolverFactory('gurobi')#, executable="/home/enzo/Downloads/gurobi10.0.2_linux64/gurobi1002/linux64/")
    opt.options['mipgap'] = 0.005

    res = opt.solve(model, tee=True)
    print(res)

    actions = [np.array([model.charging_signal[i,j].value + model.discharging_signal[i,j].value for j in range(n_evs)]) for i in range(length_time_load_pv)]
    actions = pd.DataFrame({"action": actions})

    actions.index = pd.date_range(start="2020-01-01 00:00", end="2020-12-30 23:59", freq="15T")

    actions["hid"] = actions.index.hour + actions.index.minute/60

    len_day = 24*4
    action_plot = []
    for i in range(len_day):
        action_plot.append(actions.groupby("hid").mean()["action"].reset_index(drop=True)[i].mean())
    plt.plot(action_plot)
    plt.show()

    lin_norm_vec_env.reset()
    start_time = lin_norm_vec_env.env_method("get_start_time")[0]
    end_time = pd.to_datetime(start_time) + dt.timedelta(hours=n_steps)
    env_actions = actions.loc[(actions.index >= start_time) & (actions.index <= end_time), "action"].reset_index(drop=True)

    for i in range(n_steps*time_steps_per_hour):
        lin_norm_vec_env.step([np.multiply(np.ones(n_evs), env_actions[i])])

    lin_log: pd.DataFrame = lin_norm_vec_env.env_method("get_log")[0]

    lin_log.reset_index(drop=True, inplace=True)
    lin_log = lin_log.iloc[0:-2]

    # night_log.to_csv(f"log_dumb_{use_case}_{n_evs}.csv")
    real_power_lin = []
    for i in range(lin_log.__len__()):
        lin_log.loc[i, "hour_id"] = (lin_log.loc[i, "Time"].hour + lin_log.loc[i, "Time"].minute / 60)

    mean_per_hid_lin = lin_log.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
    mean_all_lin = []
    for i in range(mean_per_hid_lin.__len__()):
        mean_all_lin.append(np.mean(mean_per_hid_lin[i]))

    mean = pd.DataFrame()
    mean["Distributed charging"] = np.multiply(mean_all_lin, 4)

    mean.plot()

    plt.xticks([0,8,16,24,32,40,48,56,64,72,80,88]
               ,["00:00","02:00","04:00","06:00","08:00","10:00","12:00","14:00","16:00","18:00","20:00","22:00"],
               rotation=45)

    plt.legend()
    plt.grid(alpha=0.2)

    plt.ylabel("Charging power in kW")
    max = lin_log.loc[0, "Observation"][-10]
    plt.ylim([-max * 1.2, max * 1.2])

    plt.show()
    