.. _agent_training:

Agent Training
============

**Configuring the environment**

Below, a basic configuration is shown for a last-mile delivery use-case.
It features multiple EVs and multiple parallel environments. New schedules will be generated in this case - this can be changed
by setting ``gen_new_sched`` to False.

.. code-block:: python

    # define fundamental parameters
    run_name = "LMD_2022_arbitrage_PPO"
    n_train_steps = 48  # number of hours in a training episode
    n_eval_steps = 8600  # number of hours in one evaluation episode
    n_eval_episodes = 1  # number of episodes for evaluation
    n_evs = 2  # number of evs
    n_envs = 2  # number of envs parallel - has to be equal to 1, if train_freq = (1, episode) or default setting
    time_steps_per_hour = 4  # temporal resolution
    use_case: str = "lmd"  # for file name
    scenario: Literal["arb", "tariff"] = "tariff"  # arbitrage or tariff
    gen_new_schedule = True  # generate a new schedule - refer to schedule generator and its config to change settings
    gen_new_test_schedule = True  # generate a new schedule for agent testing

Training parameters change training-relevant parameters in the backends of FleetRL and stable-baselines3.
This includes normalization, vectorization of environments, total training steps and the interval at which models are saved.

.. code-block:: python

    # training parameters
    norm_obs_in_env = False  # normalize observations in FleetRL (max, min normalization)
    vec_norm_obs = True  # normalize observations in SB3 (rolling normalization)
    vec_norm_rew = True  # normalize rewards in SB3 (rolling normalization)
    total_steps = int(1e6)  # total training time steps
    saving_interval = 50000  # interval for saving the model

**Creating an environment object**

To instantiate a FleetEnv object, some arguments are required, and some are optional.
This example sets the most important parameters to create an environment object for last-mile delivery.
Depending in the pricing scenario, the right input files and markups are chosen.
The same adaptation applies for the use-case's schedule and building load input files.

.. code-block:: python

    # environment arguments - adjust settings if necessary
    # additional settings can be changed in the config files
    env_kwargs = {"schedule_name": str(n_evs) + "_" + str(use_case) + ".csv",
                  "building_name": "load_" + str(use_case) + ".csv",
                  "use_case": use_case,
                  "include_building": True,
                  "include_pv": True,
                  "time_picker": "random",
                  "deg_emp": False,
                  "include_price": True,
                  "ignore_price_reward": False,
                  "ignore_invalid_penalty": False,
                  "ignore_overcharging_penalty": False,
                  "ignore_overloading_penalty": False,
                  "episode_length": n_train_steps,
                  "normalize_in_env": norm_obs_in_env,
                  "verbose": 0,
                  "aux": True,
                  "log_data": False,
                  "calculate_degradation": True,
                  "target_soc": 0.85,
                  "gen_schedule": gen_new_schedule,
                  "gen_start_date": "2022-01-01 00:00",
                  "gen_end_date": "2022-12-31 23:59:59",
                  "gen_name": "my_sched.csv",
                  "gen_n_evs": 1,
                  "seed": 42
                  }

    if scenario == "tariff":
        env_kwargs["spot_markup"] = 10
        env_kwargs["spot_mul"] = 1.5
        env_kwargs["feed_in_ded"] = 0.25
        env_kwargs["price_name"] = "spot_2021_new.csv"
        env_kwargs["tariff_name"] = "fixed_feed_in.csv"
    elif scenario == "arb":
        env_kwargs["spot_markup"] = 0
        env_kwargs["spot_mul"] = 1
        env_kwargs["feed_in_ded"] = 0
        env_kwargs["price_name"] = "spot_2021_new.csv"
        env_kwargs["tariff_name"] = "spot_2021_new_tariff.csv"

**Create environments**

Vectorized environments are created via the respective SB3 method.
The SubprocVecEnv is used because it allows for parallel processing (unlike ``DummyVecEnv``).
The VecEnv is wrapped in a normalization wrapper via ``VecNormalize``.

.. code-block:: python

    train_vec_env = make_vec_env(FleetEnv,
                                 n_envs=n_envs,
                                 vec_env_cls=SubprocVecEnv,
                                 env_kwargs=env_kwargs)

    train_norm_vec_env = VecNormalize(venv=train_vec_env,
                                      norm_obs=vec_norm_obs,
                                      norm_reward=vec_norm_rew,
                                      training=True,
                                      clip_reward=10.0)

For the validation environment, a new schedule does not need to be generated.
However, the eval time picker needs to be used, and the schedule name of the training
environment needs to be adopted.

.. code-block:: python

    env_kwargs["time_picker"] = "eval"
    env_kwargs["gen_schedule"] = False
    env_kwargs["schedule_name"] = env_kwargs["gen_name"]

    eval_vec_env = make_vec_env(FleetEnv,
                                 n_envs=n_envs,
                                 vec_env_cls=SubprocVecEnv,
                                 env_kwargs=env_kwargs)

    eval_norm_vec_env = VecNormalize(venv=eval_vec_env,
                                      norm_obs=vec_norm_obs,
                                      norm_reward=vec_norm_rew,
                                      training=True,
                                      clip_reward=10.0)

**Adding callbacks**

The SB3 EvalCallback is used to run a validation on the validation set. This tests the
currently training agent on a separate portion of the dataset, allowing for better judgement of
performance on unseen data.

.. code-block:: python

    eval_callback = EvalCallback(eval_env=eval_norm_vec_env,
                                 warn=True,
                                 verbose=1,
                                 deterministic=True,
                                 eval_freq=max(10000 // n_envs, 1),
                                 n_eval_episodes=5,
                                 render=False,
                                 )

The HyperParamCallback is used to log metric in TensorBoard. Depending on the used RL algorithm,
more parameters can be logged.

.. code-block:: python

    class HyperParamCallback(BaseCallback):
        """
        Saves hyperparameters and metrics at start of training, logging to tensorboard
        """

        def _on_training_start(self) -> None:
            hparam_dict = {
                "algorithm": self.model.__class__.__name__,
                "learning rate": self.model.learning_rate,
                "gamma": self.model.gamma,
            }

            metric_dict = {
                "rollout/ep_len_mean": 0,
                "train/value_loss": 0.0,
            }

            self.logger.record(
                "hparams",
                HParam(hparam_dict, metric_dict),
                exclude=("stdout", "log", "json", "csv")
            )

        def _on_step(self) -> bool:
            return True

    hyperparameter_callback = HyperParamCallback()

A progress bar can be included. This might not show during live training
in some remote computing environments.

.. code-block:: python

    progress_bar = ProgressBarCallback()

.. note::
    A wandb callback exists for SB3. Check the wandb documentation for implementation.

**Adding action noise, e.g. for TD3 and DDPG**

.. code-block:: python

    n_actions = train_norm_vec_env.action_space.shape[-1]
    param_noise = None
    noise_scale = 0.1
    seq_len = n_train_steps * time_steps_per_hour
    action_noise = PinkActionNoise(noise_scale, seq_len, n_actions)

**Instantiating model**

To avoid specific model tuning for each new use-case, it is recommended to
first try a model's default parameters. Below, a hyperparameter configuration is proposed for PPO
that performed well for all three use-cases. It can potentially yield potential performance
increases.

.. code-block:: python

    model = PPO(policy="MlpPolicy",
                verbose=1, # setting verbose to 0 can introduce performance increases in jupyterlab environments
                env=train_norm_vec_env,
                tensorboard_log="./tb_log")

    # might introduce performance increases
                # gamma=0.99,
                # learning_rate=0.0005,
                # batch_size=128,
                # n_epochs=8,
                # gae_lambda=0.9,
                # clip_range=0.2,
                # clip_range_vf=None,
                # normalize_advantage=True,
                # ent_coef=0.0008,
                # vf_coef=0.5,
                # max_grad_norm=0.5,
                # n_steps=2048)

Creating tensorboard instance. Port can be specified in case a certain port is free
on remote computing environments. ``Bind_all`` might be required by some remote machines.

.. code-block:: python

    %reload_ext tensorboard
    %tensorboard --logdir ./tb_log --bind_all --port 6006

Setting directories according to path names.

.. code-block:: python

    comment = run_name
    time_now = int(time.time())
    trained_agents_dir = f"./RL_agents/trained_agents/vec_PPO_{time_now}_{run_name}"
    logs_dir = f"./RL_agents/trained_agents/logs/vec_PPO_{time_now}_{run_name}"

    if not os.path.exists(trained_agents_dir):
        os.makedirs(trained_agents_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

**Model training**

Per interval, a unique model artifact is saved. Additionally, an artifact is saved
regarding the normalization metrics and the agent model - this is overwritten each time.
The instantiated callbacks must be included here.

.. code-block:: python

    # model training
    # models are saved in a specified interval: once with unique step identifiers
    # model and the normalization metrics are saved as well, overwriting the previous file every time
    for i in range(0, int(total_steps / saving_interval)):
        model.learn(total_timesteps=saving_interval,
                    reset_num_timesteps=False,
                    tb_log_name=f"PPO_{time_now}_{comment}",
                    callback=[eval_callback, hyperparameter_callback, progress_bar])

        model.save(f"{trained_agents_dir}/{saving_interval * i}")

        # Don't forget to save the VecNormalize statistics when saving the agent
        log_dir = "./RL_agents/trained_agents/tmp/vec_PPO/"
        model.save(log_dir + f"PPO-fleet_{comment}_{time_now}")
        stats_path = os.path.join(log_dir, f"vec_normalize-{comment}_{time_now}.pkl")
        train_norm_vec_env.save(stats_path)
