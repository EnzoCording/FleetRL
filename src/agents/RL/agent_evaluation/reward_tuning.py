from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from FleetRL.fleet_env.fleet_environment import FleetEnv

if __name__ == "__main__":

    eval_vec_env = make_vec_env(FleetEnv,
                                n_envs=1,
                                vec_env_cls=SubprocVecEnv,
                                env_kwargs={
                                    #"schedule_name": "reward_tuning_schedule.csv",
                                    #"building_name": "reward_tuning_load.csv",
                                    "include_building": True,
                                    "include_pv": True,
                                    "static_time_picker": False,
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
                                    "use_case": "lmd",
                                })

    eval_norm_vec_env = VecNormalize(venv=eval_vec_env,
                                     norm_obs=True,
                                     norm_reward=True,
                                     training=True,
                                     clip_reward=25)

    model = PPO.load("./../trained/new_ppo/new_PPO_1320000", env=eval_norm_vec_env)
    # model = PPO(env=eval_norm_vec_env, policy='MlpPolicy')

    print("########### \n Dumb Charging \n ##########")
    model.env.reset()
    n_steps = 48*4
    for i in range(n_steps):
        action = []
        for n in range(model.env.get_attr("num_cars")[0]):
            action.append(0)
        model.env.step([action])

    mean, _ = evaluate_policy(env=eval_norm_vec_env, n_eval_episodes=1, deterministic=True, model=model)
    print(mean)