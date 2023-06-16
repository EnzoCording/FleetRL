import optuna
import gymnasium as gym
from stable_baselines3 import TD3
import FleetRL

fleet_env = gym.make('FleetEnv-v0',
               schedule_name="lmd_sched_single.csv",
               building_name="load_lmd.csv",
               include_building=True,
               include_pv=True,
               static_time_picker=False,
               deg_emp=False,
               include_price=True,
               ignore_price_reward=False,
               ignore_invalid_penalty=False,
               ignore_overcharging_penalty=False,
               ignore_overloading_penalty=False,
               episode_length=144)

def objective(trial):
    env = fleet_env
    gamma = trial.suggest_uniform('gamma', 0.9, 0.99)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    buffer_size = trial.suggest_loguniform('buffer_size', 50000, 1000000)
    batch_size = trial.suggest_uniform('batch_size', 32, 1024)
    tau = trial.suggest_uniform('tau', 0.001, 0.01)

    model = TD3('MlpPolicy',
            env,
            verbose=0,
            learning_rate=learning_rate,
            learning_starts=2000,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
           )

    model.learn(50000)

    mean_reward = evaluate(model, env, n_episodes=5)
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
