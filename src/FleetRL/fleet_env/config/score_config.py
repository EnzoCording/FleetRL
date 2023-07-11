class ScoreConfig:

    def __init__(self):
        self.price_multiplier = 3
        self.price_exponent = 1
        self.penalty_soc_violation = -750
        self.clip_soc_violation = -1000
        self.penalty_overloading = -150
        self.clip_overloading = -500
        # in cold weather trafo can operate >100%
        # TODO give overcharging/underloading penalty not only when the car departs, but
        # also in realtime when the battery dips below or goes above the healthy range
        # And maybe increase the penalty the farther the agent gets away from these boundaries
        self.penalty_invalid_action = -0.05
        self.penalty_overcharging = -0.015
        self.clip_overcharging = -0.05

        # possible reward: money spent/earned due to buying/selling electricity for charging/discharging
        # but these rewards fluctuate at runtime and so are not configured here explicitly
