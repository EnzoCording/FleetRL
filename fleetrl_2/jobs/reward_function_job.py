import logging

from tidysci.package_manager import PackageManager
from tidysci.task import Task
from tidysci.task import register

logger = logging.getLogger(PackageManager.get_name())


@register(alias=True)
class RewardFunctionJob(Task):

    def __init__(self,
                 ignore: dict,
                 penalty: dict,
                 reward: dict,
                 price_multiplier,
                 price_exponent,
                 _dir_root: str,
                 rng_seed: int
                 ):
        super().__init__(_dir_root, rng_seed)

        self.price_multiplier = price_multiplier
        self.price_exponent = price_exponent
        self.ignore = _Ignore(**ignore)
        self.penalty = _Penalty(**penalty)
        self.reward = _Reward(**reward)

    def is_finished(self) -> bool:
        return True


class _Reward:
    def __init__(self,
                 fully_charged: int = 1):
        self.fully_charged = fully_charged


class _Penalty:
    def __init__(self,
                 invalid_action=-0.2,
                 overcharging=-0.0055,
                 overloading_multiplier=1,
                 clip_overcharging=-0.2
                 ):
        self.invalid_action = invalid_action
        self.overcharging = overcharging
        self.overloading_multiplier = overloading_multiplier
        self.clip_overcharging = clip_overcharging


class _Ignore:
    def __init__(self,
                 invalid_action=False,
                 overcharging=False,
                 overloading=False,
                 price=False
                 ):
        self.invalid_action = invalid_action
        self.overcharging = overcharging
        self.overloading = overloading
        self.price = price
