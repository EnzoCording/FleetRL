.. _benchmarking:

Benchmarking
=============

Below, the logic of the three static benchmarks is presented, along with the linear optimization model.
On github, fully implemented benchmarking notebooks are uploaded for each benchmark.

**Uncontrolled charging**

Every time step, the maximum charging signal is sent so that the battery is immediately charged
upon arrival.

.. code-block:: python

    for i in range(episode_length * timesteps_per_hour * n_episodes):
        if dumb_norm_vec_env.env_method("is_done")[0]:
            dumb_norm_vec_env.reset()
        dumb_norm_vec_env.step([np.ones(n_evs)])

**Distributed charging**

In the distributed charging strategy, the charging process is spread out across the entire stay,
and the battery only reaches full charge before arrival. The method ``get_dist_factor`` is used to
get the laxity / distribution factor for each time step: time needed / time left.

.. code-block:: python

    for i in range(episode_length * timesteps_per_hour * n_episodes):
    if dist_norm_vec_env.env_method("is_done")[0]:
        dist_norm_vec_env.reset()
    dist_norm_vec_env.step(([np.clip(np.multiply(np.ones(n_evs), dist_norm_vec_env.env_method("get_dist_factor")[0]),0,1)]))

**Night charging**

Night charging preferably starts charging at midnight, but begins earlier if the battery requires a longer
charging time. The earliest departure time globally is considered to ensure a full battery in the worst
case scenario. During the day between 11 and 14, a charging signal is sent to the battery to accommodate for the
lunch break in the caretaker use-case.

.. code-block:: python

    df = env.db
    df_leaving_home = df[(df['Location'].shift() == 'home') & (df['Location'] == 'driving')]
    earliest_dep_time = df_leaving_home['date'].dt.time.min()
    day_of_earliest_dep = df_leaving_home[df_leaving_home['date'].dt.time == earliest_dep_time]['date'].min()
    earliest_dep = earliest_dep_time.hour + earliest_dep_time.minute/60

    evse = env.load_calculation.evse_max_power
    cap = env.ev_conf.init_battery_cap
    target_soc = env.ev_conf.target_soc
    eff = env.ev_conf.charging_eff

    max_time_needed = target_soc * cap / eff / evse  # time needed to charge to target soc from 0
    difference = earliest_dep - max_time_needed
    starting_time = (24 + difference)
    if starting_time > 24:
        starting_time = 23.99 # always start just before midnight

    charging_hour = int(math.modf(starting_time)[1])
    minutes = np.asarray([0, 15, 30, 45])
    # split number and decimals, use decimals and choose the closest minute
    closest_index = np.abs(minutes - int(math.modf(starting_time)[0]*60)).argmin()
    charging_minute = minutes[closest_index]

    episode_length = n_steps
    n_episodes = n_episodes
    night_norm_vec_env.reset()

    charging=False

    for i in range(episode_length * timesteps_per_hour * n_episodes):
        if night_norm_vec_env.env_method("is_done")[0]:
            night_norm_vec_env.reset()
        time: pd.Timestamp = night_norm_vec_env.env_method("get_time")[0]
        if ((time.hour >= 11) and (time.hour <= 14)) and (use_case=="ct"):
            night_norm_vec_env.step(([np.clip(np.multiply(np.ones(n_evs), night_norm_vec_env.env_method("get_dist_factor")[0]),0,1)]))
            continue
        time: pd.Timestamp = night_norm_vec_env.env_method("get_time")[0]
        if (((charging_hour <= time.hour) and (charging_minute <= time.minute)) or (charging)):
            if not charging:
                charging_start: pd.Timestamp = copy(time)
            charging=True
            night_norm_vec_env.step([np.ones(n_evs)])
        else:
            night_norm_vec_env.step([np.zeros(n_evs)])
        if charging and ((time - charging_start).total_seconds()/3600 > int(max_time_needed)):
            charging = False

**Linear optimization model**

The linear optimization model does 2 things: first, it calculates an optimal result, based on the available
information of the FleetRL use-case (building load, vehicle schedules, etc.). Its objective is cost minimization and its variables are the
actions in the range of [0,1] -> identical to the RL agents. Once a result has been obtained, it is run on a
FleetRL environment -> the linear agent steps through the env and generates the log file that saves charging cost,
SoH, violations, etc.

.. code-block:: python

    # importing necessary libraries
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

    # wrapper for parallelization
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

        # adapting price information according to user input
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

        # creating vec_env for linear agent
        lin_vec_env = make_vec_env(FleetEnv,
                                   n_envs=n_envs,
                                   vec_env_cls=SubprocVecEnv,
                                   env_kwargs=env_kwargs)

        # normalization
        lin_norm_vec_env = VecNormalize(venv=lin_vec_env,
                                        norm_obs=True,
                                        norm_reward=True,
                                        training=True,
                                        clip_reward=10.0)

        # creating an env object for accessing information in the pyomo model
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

        # length of time, load, and pv series (with multiple EVs, this is the index)
        length_time_load_pv = 8760 * 4

        # price and tariff data
        price_data = np.multiply(np.add(df["DELU"], env.ev_conf.fixed_markup), env.ev_conf.variable_multiplier) / 1000
        tariff_data = np.multiply(df["tariff"], 1-env.ev_conf.feed_in_deduction) / 1000

        # pv data
        pv_data = df["pv"]  # pv power in kW

        # soc on return, separately for each EV
        soc_on_return = [df.loc[df["ID"]==i, "SOC_on_return"] for i in range(n_evs)]

        # further model parameters
        battery_capacity = env.ev_conf.init_battery_cap  # EV batt size in kWh
        p_trafo = env.load_calculation.grid_connection  # Transformer rating in kW
        charging_eff = env.ev_conf.charging_eff  # charging losses
        discharging_eff = env.ev_conf.discharging_eff  # discharging losses
        init_soc = env.ev_conf.def_soc  # init SoC
        evse_max_power = env.load_calculation.evse_max_power  # kW, max rating of the charger

        # create pyomo model
        model = pyo.ConcreteModel(name="sc_pyomo")

        # create sets for the pyomo optimization
        model.timestep = pyo.Set(initialize=range(length_time_load_pv))
        model.time_batt = pyo.Set(initialize=range(0, length_time_load_pv+1))
        model.ev_id = pyo.Set(initialize=range(n_evs))

        # model parameters, i,j used in case multiple EVs exist
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

        # constraints
        def grid_limit(m, i, ev):
            return ((m.charging_signal[i, ev] + m.discharging_signal[i, ev]) * evse_max_power
                    + m.building_load[i] - m.pv[i] <= p_trafo)

        # charging an discharging cannot occur at the same time, big M method
        def mutual_exclusivity_charging(m, i, ev):
            return m.charging_signal[i, ev] <= m.positive_action[i, ev]

        def mutual_exclusivity_discharging(m, i, ev):
            return m.discharging_signal[i, ev] >= (m.positive_action[i, ev] - 1)

        # PV prioritized over grid
        def pv_use(m, i, ev):
            return m.used_pv[i, ev] <= m.charging_signal[i, ev] * evse_max_power

        # only use as much PV as available
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

        # cost minimization
        def obj_fun(m):
            return (sum([((m.charging_signal[i, ev] * evse_max_power - m.used_pv[i, ev]) / time_steps_per_hour) * m.price[i] +
                         ((m.discharging_signal[i, ev] * evse_max_power * discharging_eff) / time_steps_per_hour) * m.tariff[i]
                         for i in m.timestep for ev in range(n_evs)]))

        # change solver here to glpk if gurobi not configured
        model.obj = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
        opt = pyo.SolverFactory('gurobi')#, executable="/home/enzo/Downloads/gurobi10.0.2_linux64/gurobi1002/linux64/")
        # for quicker solving
        opt.options['mipgap'] = 0.005
        # print additional information
        res = opt.solve(model, tee=True)
        print(res)

        # extract actions array for each time step, this is the result of the optimization
        actions = [np.array([model.charging_signal[i,j].value + model.discharging_signal[i,j].value for j in range(n_evs)]) for i in range(length_time_load_pv)]
        actions = pd.DataFrame({"action": actions})

        # set the same index as for the RL agent and the other benchmarks
        actions.index = pd.date_range(start="2020-01-01 00:00", end="2020-12-30 23:59", freq="15T")
        actions["hid"] = actions.index.hour + actions.index.minute/60

        # plot the resulting action curve from the pyomo optimization
        len_day = 24*4
        action_plot = []
        for i in range(len_day):
            action_plot.append(actions.groupby("hid").mean()["action"].reset_index(drop=True)[i].mean())
        plt.plot(action_plot)
        plt.show()

        # feed the resulting actions into the FleetRL environment to get log data and KPIs
        lin_norm_vec_env.reset()
        start_time = lin_norm_vec_env.env_method("get_start_time")[0]
        end_time = pd.to_datetime(start_time) + dt.timedelta(hours=n_steps)
        env_actions = actions.loc[(actions.index >= start_time) & (actions.index <= end_time), "action"].reset_index(drop=True)

        for i in range(n_steps*time_steps_per_hour):
            lin_norm_vec_env.step([np.multiply(np.ones(n_evs), env_actions[i])])

        # get log from environment
        lin_log: pd.DataFrame = lin_norm_vec_env.env_method("get_log")[0]

        # plotting analog to RL evaluation
        lin_log.reset_index(drop=True, inplace=True)
        lin_log = lin_log.iloc[0:-2]

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

        # save log as pickle
        lin_log.to_pickle("lin_log.pickle")
