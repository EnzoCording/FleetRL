from gym.envs.registration import registry, register, make, spec

register(
     id='FleetEnv-v0',
     entry_point='FleetRL.env:FleetEnv',
     max_episode_steps=200,
)
