.. _custom_env:

Creating a custom environment
=============================

This section walks through the process of creating a completely custom-made environment, potentially to model your
own commercial use-case and train an RL agent on it.

.. note::

    Emphasis was laid on keeping the amount of coding to a minimum for running a custom use-case.

**The different components**

To build a custom use-case in FleetRL, an understanding of the different building blocks of the framework is required.
Inputs come in different formats, e.g. time-series, static inputs, and config files. Below, each component is explained
and how to modify it.

Time-series
-----------

**Vehicle schedules**

Vehicle schedules are the most important time-series required in FleetRL. They are saved as csv files. Agents can be trained solely with schedules,
if price, PV, and building load are set to be ignored. The schedule format is adapted from emobpy, an open-source
framework for EV schedule generation, to allow for cross-compatibility with emobpy outputs. The following columns are
required:

 * **date**: The "date" column displays the date, preferably in the format ``YYYY-MM-dd hh:mm:ss``. By default, the time
   resolution is 15 minutes in the schedules - a lower time resolution can later be adjusted in the settings.
 * **Distance_km**: Distance travelled in each time step, in km
 * **Consumption_kWh**: Energy consumed in each time step, in kWh
 * **Location**: Location specifier. "home" and "driving" must always be included.
 * **ChargingStation**: Specifying the type of charging station. "home" and "none" must always be included. A
   ChargingStation of "none" means that the car is "driving".
 * **ID**: The vehicle ID. Starts with 0 and goes until (N-1), if N cars are present in the schedule.
 * **PowerRating_kW**: The amount of kW at the charger.

.. note::
    The "date" column in the schedule csv file dictates the timeframe of the entire simulation.

.. note::
    Custom schedules can be generated within FleetRL. It currently is possible to generate two mobility scenarios:
    departure - arrival, and departure - break (arrival) - break (departure) - arrival. A fully automated process exists,
    which is explained in the next section - custom schedules are thereby generated when creating the environment
    object for the first time. To customize the statistics of the vehicle schedules, the ``schedule_config.py`` file
    must be modified. It specifies the behavioral statistics: arrival/departure times (mean, standard dev), energy
    consumption (mean, standard dev), distance (mean, standard dev), and differences between weekdays, saturday, and
    sunday.

**Electricity prices**

Electricity prices must be csv files and have the same date range as the vehicle schedules. The backend of FleetRL is
currently set to German spot prices that can be downloaded from smard.de or the Entso-E transparency platform. The
following columns are required:

 * **date**: Date, preferably in ``YYYY-MM-dd hh:mm:ss`` format. If time resolutions of lower than 15 minutes are provided,
   FleetRL automatically upsamples the data to match the time resolution of the vehicle schedules. German spot prices
   have a time resolution of 1 hour and are thus upscaled to 15 minutes.
 * **Deutschland/Luxemburg [€/MWh] Original resolutions**: This is the identifier for the German/Lux electricity market
   on ENTSO-E. The name of the price column can also be changed in the ``data_processing.load_prices`` method.

.. note::
    A csv file for buying ("price") and selling ("tariff") is required. The tariff csv must have a "tariff" column,
    where the prices are specified.

**Building load**

The TMY-3 dataset was used for building load time series. Different commercial buildings exist,
such as "warehouse", "office", "hospital", etc. The state of Washington was used due to its resemblance with the
German climate. Following columns must be present:

 * **date**: Preferably ``YYYY-MM-dd hh:mm:ss`` format, same range as the vehicle schedules. Varying time resolutions are
   automatically adjusted.
 * **load**: The "load" column specifies the load in kW in the current time step
 * **pv**: PV generation can be specified in a "pv" column. Alternatively, PV generation can be input via a separate CSV
   file. In that case, it must have a "date" and "pv" column.

Static inputs
-------------

**Use-case-specific inputs**

 * Number of vehicles: This is specified via the number of vehicles present in the schedule file (taking the max of
   the "ID" column).
 * Vehicle type: Battery capacity must be specified in ``fleet_environment.__init__`` and ``episode.load_calculation``.
   In ``config.ev_config``, further parameters can be changed, such as onboard charger capacity, charging efficiency,
   etc. EVSE (EV charger) capacity is specified under ``episode.load_calculation``.
 * Grid connection: The grid connection is specified in ``episode.load_calculation``. It can either be hard-coded, or
   adjusted depending on the number of cars and the maximum building load (e.g. last-mile delivery).
 * Electricity tax, markups, fees: Can be adjusted when creating an env object, or under ``config.ev_config``.

**Episode-specific inputs**
 * Episode length: In hours, can be adjusted when creating an env object.
 * Time resolution: Hourly ("1H") or 15 minutes ("15T"). Can be set under ``config.time_config``.
 * Auxiliary inputs: Can help the agent to solve the task by providing additional information and preprocesses metrics.

**Reward function and penalties**

``config.score_config`` specifies the shape, scale and type of rewards and penalties. A reward signal exists for the
following:

 * Charging expenses: The agent receives a reward signal based on the money spent on EV charging
 * Overloading the grid connection: A penalty signal is sent if the grid connection exceeds the nominal trafo rating.
   Its shape follows the sigmoid function: small exceeding values (0-10%) can be handled by a trafo and thus do not
   require a strong penalty. Higher values (15-50%) must be avoided and thus require strong penalties. Everything above
   most likely represents a system failure, and a further increase does not worsen the situation - the level of penalties
   thus flattens off towards high overloadings.
 * SOC violation: If a car leaves without being fully charged, a similarly shaped penalty signal is sent to the agent.
   It too is sigmoid shaped.
 * Battery overcharging: If the agent sends a signal that would overcharge / over-discharge the battery, a penalty
   signal can be sent.
 * Invalid action: If a charging signal is sent to an empty charging spot, a penalty can be sent to the agent.

Summary
-------

When creating a custom commercial use-case, do the following:

**Environment object constructor**

Customizable parameters can be found below:

.. code-block:: python

    # environment arguments - adjust settings if necessary
    # additional settings can be changed in the config files
    env_kwargs = {"schedule_name": str(n_evs) + "_" + str(use_case) + ".csv",  # schedule name
                  "building_name": "load_" + str(use_case) + ".csv",  # name of building load (and pv) time series
                  "use_case": use_case,  # put "custom" here, or your own identifier (code modifications necessary)
                  "include_building": True,  # including building load in the observations
                  "include_pv": True,  # including PV in the observations
                  "time_picker": "random",  # randomly picking a starting time from the training set
                  "deg_emp": False,  # not using empirical degradation (uses non-linear degradation model)
                  "include_price": True,  # including prices in the observations
                  "ignore_price_reward": False,  # True: not sending charging cost reward signals to the agent
                  "ignore_invalid_penalty": False,  # True: not sending invalid action penalties to the agent
                  "ignore_overcharging_penalty": False,  # True not sending overcharging penalties to the agent
                  "ignore_overloading_penalty": False,  # True not sending overloading penalties to the agent
                  "episode_length": n_train_steps,  # Episode length in hours
                  "normalize_in_env": norm_obs_in_env,  # Normalize observations within FleetRL (min/max normalization)
                  "verbose": 0,  # print statements, can slow down FPS
                  "aux": True,  # Include auxiliary information in observation
                  "log_data": False,  # log data in a dataframe (best used in evaluation, no need during training)
                  "calculate_degradation": True,  # call the degradation class (can slow down FPS)
                  "target_soc": 0.85,  # target SOC
                  "gen_schedule": gen_new_schedule,  # generate a new schedule upon env object creation
                  "gen_start_date": "2022-01-01 00:00",  # start date of the new schedule
                  "gen_end_date": "2022-12-31 23:59:59",  # end date of the new schedule
                  "gen_name": "my_sched.csv",  # name of the generated schedule
                  "gen_n_evs": 1,  # number of EVs to be generated (approx. 20 min. per EV)
                  "seed": 42  # RNG seed for the generation
                  }

    if scenario == "tariff":  # if tariff is chosen, price and tariff are different, there are mark-ups and deductions
        env_kwargs["spot_markup"] = 10
        env_kwargs["spot_mul"] = 1.5
        env_kwargs["feed_in_ded"] = 0.25
        env_kwargs["price_name"] = "spot_2021_new.csv"
        env_kwargs["tariff_name"] = "fixed_feed_in.csv"
    elif scenario == "arb":  # if arb is chosen, price and tariff are the same, no mark-ups or deductions
        env_kwargs["spot_markup"] = 0
        env_kwargs["spot_mul"] = 1
        env_kwargs["feed_in_ded"] = 0
        env_kwargs["price_name"] = "spot_2021_new.csv"
        env_kwargs["tariff_name"] = "spot_2021_new_tariff.csv"

**Schedule config**

Configure the statistics in ``schedule.schedule_config``. View the "custom" case.

**Load calculation module**

Configure the battery capacity, EVSE power and grid limit in ``load_calculation.load_calculation``. View the custom case.

**Other configs**

Modify the parameters in ``fleet_env.config.ev_config``, ``fleet_env.config.score_config``, and
``fleet_env.config.time_config``.