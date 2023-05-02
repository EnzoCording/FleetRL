from gymnasium import register

register(
     id='FleetEnv-v0',
     entry_point='FleetRL.fleet_env:FleetEnv',
     max_episode_steps=None,
)
