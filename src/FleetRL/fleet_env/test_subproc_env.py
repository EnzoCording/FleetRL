import FleetRL
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import numpy as np
from stable_baselines3 import TD3


def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id, schedule_name="lmd_sched_single.csv",
                       building_name="load_lmd.csv",
                       include_building=False,
                       include_pv=False,
                       include_price=True,
                       static_time_picker=False,
                       target_soc=0.85,
                       init_soh=1.0,
                       deg_emp=False,
                       ignore_price_reward=False,
                       ignore_overloading_penalty=False,
                       ignore_invalid_penalty=False,
                       ignore_overcharging_penalty=False,
                       episode_length=24,
                       log_to_csv=False,
                       calculate_degradation=False,
                       verbose=1,
                       normalize_in_env=False)

        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    env_id = "FleetEnv-v0"
    num_cpu = 4
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    vec_env = VecNormalize(venv=vec_env, training=True, norm_obs=True, norm_reward=True)

    model = TD3("MlpPolicy", vec_env, verbose=1, train_freq=4)
    model.learn(total_timesteps=250)

    obs = vec_env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        print(rewards)
        vec_env.render()

    make_vec_env(env_id="FleetEnv-v0", vec_env_cls=SubprocVecEnv, vec_env_kwargs={
        "schedule_name": "lmd_sched_single.csv",
        "building_name": "load_lmd.csv",
        "include_building": False,
        "include_pv": False,
        "static_time_picker": False,
        "deg_emp": False,
        "include_price": False,
        "ignore_price_reward": True,
        "ignore_invalid_penalty": False,
        "ignore_overcharging_penalty": False,
        "ignore_overloading_penalty": True,
        "episode_length": 36,
        "verbose": 0,
        "calculate_degradation": False
    })
