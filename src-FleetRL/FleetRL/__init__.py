from gym.envs.registration import register

register(
     id='FleetEnv-v0',
     entry_point='FleetRL.env:FleetEnv',
     max_episode_steps=None,
)
