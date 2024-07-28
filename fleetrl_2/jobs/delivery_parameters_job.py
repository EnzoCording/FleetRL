from tomlchef.job import registered_job

from .schedule_parameters_job import ScheduleParametersJob


@registered_job
class DeliveryParametersJob(ScheduleParametersJob):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_toml_key() -> str:
        return "delivery_parameters"

    def is_finished(self) -> bool:
        return True
