class ScoreConfig:

    def __init__(self):
        self.price_multiplier = 1000.0
        self.penalty_soc_violation = -500.0
        self.penalty_overloading = -50.0
        # in cold weather trafo can operate >100%
        # TODO give overcharging/underloading penalty not only when the car departs, but
        # also in realtime when the battery dips below or goes above the healthy range
        # And maybe increase the penalty the farther the agent gets away from these boundaries
        self.penalty_invalid_action = -0.15
        self.penalty_overcharging = -0.05

        # possible reward: money spent/earned due to buying/selling electricity for charging/discharging
        # but these rewards fluctuate at runtime and so are not configured here explicitly
