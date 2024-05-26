from pyjob_todo.job import Job
from pyjob_todo.job import registered_job


@registered_job
class SiteParametersJob(Job):

    def __init__(self,
                 max_grid_connection: float,
                 fixed_markup: float,
                 variable_multiplier: float,
                 feed_in_deduction: float,
                 data: dict,
                 **kwargs):

        super.__init__(**kwargs)
        self.max_grid_connection = max_grid_connection
        self.fixed_markup = fixed_markup
        self.variable_multiplier = variable_multiplier
        self.feed_in_deduction = feed_in_deduction

        self._external_electricity_price = data["_external_electricity_price"]
        self._external_feed_in_tariff = data["_external_feed_in_tariff"]

    @staticmethod
    def get_toml_key() -> str:
        return "site_parameters"

    def is_finished(self) -> bool:
        return True
