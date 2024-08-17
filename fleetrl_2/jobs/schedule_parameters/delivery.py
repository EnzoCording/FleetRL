from tidysci.task import register

from fleetrl_2.jobs.schedule_parameters.schedule_parameters import \
    ScheduleParameters


@register(alias=True)
class Delivery(ScheduleParameters):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_finished(self) -> bool:
        return True
