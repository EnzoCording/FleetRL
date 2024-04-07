import numpy as np

class ScoreConfig:
    """
    The Score Config sets coefficients to calculate the reward function
    - Multipliers: price_multiplier, penalty_invalid_action, penalty_overcharging, penalty_overloading
    - Changing the multipliers to 0 leads to ignoring this aspect of the reward function
    - The SOC violation and overloading are calculated using sigmoid functions
    """

    def __init__(self, env_config):
        self.price_multiplier = env_config.get('price_multiplier', 3.33)
        self.price_exponent = env_config.get('price_exponent', 1)
        self.fully_charged_reward = env_config.get('fully_charged_reward', 1)
        # in cold weather trafo can operate >100%
        # TODO give overcharging/underloading penalty not only when the car departs, but
        # also in realtime when the battery dips below or goes above the healthy range
        # And maybe increase the penalty the farther the agent gets away from these boundaries
        self.penalty_invalid_action = env_config.get('penalty_invalid_action', -0.2)
        self.penalty_overcharging = env_config.get('penalty_overcharging', -0.0055)
        self.penalty_overloading = env_config.get('penalty_overloading', 1)
        self.clip_overcharging = env_config.get('clip_overcharging', -0.2)

    @staticmethod
    # Define the soc_violation_penalty function using the parameters of the fitted sigmoid function for the reward function
    def soc_violation_penalty(missing_soc):
        x0, k = 0.29229767, 16.48461585  # Parameters from the fitted sigmoid function for the reward function
        penalty = -500 / (1 + np.exp(-k * (missing_soc - x0))) + 1

        return penalty

    # Define the overloading_penalty function using the parameters of the fitted piecewise sigmoid function for the overloading penalty function
    def overloading_penalty(self, rel_loading):
        x0, k = 1.33298382, 15.77350877  # Parameters from the fitted piecewise sigmoid function for the overloading penalty function
        penalty = np.piecewise(rel_loading, [rel_loading < 1.1, rel_loading >= 1.1],
                               [0, lambda x: -700 / (1 + np.exp(-k * (x - x0)))])
        # If penalty is an array with only one element, return that element
        if isinstance(penalty, np.ndarray) and penalty.size == 1:
            penalty = penalty.item()

        return penalty * self.penalty_overloading

