from tomlchef.job import registered_job

from .schedule_statistics_job import ScheduleStatisticsJob


@registered_job
class DeliveryStatisticsJob(ScheduleStatisticsJob):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_toml_key() -> str:
        return "delivery_statistics"

    def is_finished(self) -> bool:
        return True
