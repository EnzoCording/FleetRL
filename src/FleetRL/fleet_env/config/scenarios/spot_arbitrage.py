from FleetRL.fleet_env.fleet_environment import FleetEnv

env = FleetEnv(schedule_name="lmd_sched_single",
               building_name="load_lmd",
               price_name="spot_2021_new.csv",
               tariff_name="spot_2021_new.csv",
               use_case="lmd",
               verbose=False,
               time_picker="random",
               episode_length=48,
               calculate_degradation=True,
               log_data=False,
               normalize_in_env=False,
               aux=True,
               spot_markup=0,
               spot_mul=1,
               feed_in_ded=0
               )

env_args = {"schedule_name": "lmd_sched_single",
            "building_name": "load_lmd",
            "price_name": "spot_2021_new.csv",
            "tariff_name": "spot_2021_new.csv",
            "use_case": "lmd",
            "verbose": False,
            "time_picker": "random",
            "episode_length": 48,
            "calculate_degradation": True,
            "log_data": False,
            "normalize_in_env": False,
            "aux": True,
            "spot_markup": 0,
            "spot_mul": 1,
            "feed_in_ded": 0
            }
