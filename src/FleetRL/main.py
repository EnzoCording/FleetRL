from FleetRL.fleet_env.fleet_environment import FleetEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import TD3, PPO, SAC

# env = FleetEnv(episode_length=48,
#                verbose=True,
#                gen_schedule=True,
#                gen_n_evs=2,
#                gen_start_date="2020-01-01 00:00:00",
#                gen_end_date="2020-01-02 23:59:00",
#                gen_name="my_schedule.csv",
#                use_case="lmd",
#                building_name="load_lmd.csv",
#                include_pv=True,
#                include_building=True,
#                include_price=True,
#                aux=True,
#                ignore_price_reward=False,
#                ignore_invalid_penalty=False,
#                ignore_overcharging_penalty=False,
#                ignore_overloading_penalty=False,
#                init_soh=1.0,
#                target_soc=0.95,
#                time_picker="random",
#                deg_emp=False,
#                calculate_degradation=True,
#                normalize_in_env=False
#                )

train_vec_env_args_dict = {"episode_length": 48,
                           "verbose": True,
                           "gen_schedule": True,
                           "gen_n_evs": 2,
                           "gen_start_date": "2020-01-01 00:00:00",
                           "gen_end_date": "2020-01-02 23:59:00",
                           "gen_name": "my_schedule.csv",
                           "use_case": "lmd",
                           "building_name": "load_lmd.csv",
                           "include_pv": True,
                           "include_building": True,
                           "include_price": True,
                           "aux": True,
                           "ignore_price_reward": False,
                           "ignore_invalid_penalty": False,
                           "ignore_overcharging_penalty": False,
                           "ignore_overloading_penalty": False,
                           "init_soh": 1.0,
                           "target_soc": 0.95,
                           "time_picker": "random",
                           "deg_emp": False,
                           "calculate_degradation": True,
                           "normalize_in_env": False}

eval_vec_env_args_dict = train_vec_env_args_dict.copy()
eval_vec_env_args_dict["time_picker"] = "eval"
eval_vec_env_args_dict["verbose"] = False

n_envs = 2

train_vec_env = make_vec_env(env_id=FleetEnv, env_kwargs=train_vec_env_args_dict, vec_env_cls=SubprocVecEnv)
train_norm_vec_env = VecNormalize(venv=train_vec_env, norm_obs=True, norm_reward=True)

eval_vec_env = make_vec_env(env_id=FleetEnv, env_kwargs=eval_vec_env_args_dict, vec_env_cls=SubprocVecEnv)
eval_norm_vec_env = VecNormalize(venv=eval_vec_env, norm_obs=True, norm_reward=True)

eval_callback = EvalCallback(eval_env=eval_norm_vec_env,
                             warn=True,
                             verbose=1,
                             deterministic=True,
                             eval_freq=max(10000 // n_envs, 1),
                             n_eval_episodes=5,
                             render=False,
                             )
