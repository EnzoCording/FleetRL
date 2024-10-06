from pathlib import Path

from tidysci.task import Task
from tidysci.task import register


@register(alias=True)
class SiteParametersJob(Task):

    def __init__(self,
                 max_grid_connection: float,
                 fixed_markup: float,
                 variable_multiplier: float,
                 feed_in_deduction: float,
                 price_data: dict,
                 _dir_root: str,
                 rng_seed: int):

        super().__init__(_dir_root, rng_seed)
        self.max_grid_connection = max_grid_connection
        self.fixed_markup = fixed_markup
        self.variable_multiplier = variable_multiplier
        self.feed_in_deduction = feed_in_deduction

        self.site_data = _SiteData(**price_data)

        self._external_electricity_price = price_data["_external_electricity_price"]
        self._external_feed_in_tariff = price_data["_external_feed_in_tariff"]

class _SiteData():
    def __init__(self,
                 external_electricity_price: str | None = None,
                 external_feed_in_tariff: str | None = None):

        self.external_electricity_price = Path(
            external_electricity_price) if external_electricity_price is not None else None
        self.external_feed_in_tariff = Path(
            external_feed_in_tariff) if external_feed_in_tariff is not None else None

    def is_finished(self) -> bool:
        return True
