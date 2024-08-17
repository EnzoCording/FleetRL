from tidysci.task import Task
from tidysci.task import register


@register(alias=True)
class SiteParametersJob(Task):

    def __init__(self,
                 max_grid_connection: float,
                 fixed_markup: float,
                 variable_multiplier: float,
                 feed_in_deduction: float,
                 data: dict,
                 _dir_root: str,
                 rng_seed: int):
        super().__init__(_dir_root, rng_seed)
        self.max_grid_connection = max_grid_connection
        self.fixed_markup = fixed_markup
        self.variable_multiplier = variable_multiplier
        self.feed_in_deduction = feed_in_deduction

        self._external_electricity_price = data["_external_electricity_price"]
        self._external_feed_in_tariff = data["_external_feed_in_tariff"]

    def is_finished(self) -> bool:
        return True
