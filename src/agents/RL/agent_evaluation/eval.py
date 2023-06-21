from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from FleetRL.fleet_env.fleet_environment import FleetEnv

if __name__ == "__main__":

    eval_vec_env = make_vec_env(FleetEnv,
                                n_envs=1,
                                vec_env_cls=SubprocVecEnv,
                                env_kwargs={
                                    "schedule_name": "lmd_sched_2.csv",
                                    "building_name": "load_lmd.csv",
                                    "include_building": True,
                                    "include_pv": True,
                                    "eval_time_picker": True,
                                    "deg_emp": False,
                                    "include_price": True,
                                    "ignore_price_reward": False,
                                    "ignore_invalid_penalty": False,
                                    "ignore_overcharging_penalty": False,
                                    "ignore_overloading_penalty": False,
                                    "episode_length": 48,
                                    "normalize_in_env": False,
                                    "verbose": 1,
                                    "aux": True,
                                    "use_case": "lmd"
                                })

    eval_norm_vec_env = VecNormalize(venv=eval_vec_env,
                                     norm_obs=True,
                                     norm_reward=True,
                                     training=True,
                                     clip_reward=10.0)

    # model = TD3.load("./../trained/td3_aux_reworder_260000/td3_aux_reworder_260000", env=eval_norm_vec_env)
    # model = PPO.load("./../trained/vec_ppo-1687255378-ppo_full_vecnorm_clip5_aux_rewardshape_order_2006_12/820000.zip", env=eval_norm_vec_env)
    model = TD3.load("./../trained/td3_aux_lr001_2cars_run3_harsher_3rew/980000", env = eval_norm_vec_env)
    # model = TD3.load("./../trained/vec_TD3-1687291761-td3_2_cars_run2/260000", env = eval_norm_vec_env)

    mean_reward, _ = evaluate_policy(model, eval_norm_vec_env, n_eval_episodes=5, deterministic=True)
    print(mean_reward)
