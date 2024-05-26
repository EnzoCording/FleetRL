from pyjob_todo.job import Job
from pyjob_todo.job import registered_job

class DepartureTime:
    def __init__(self,
                 dep_mean_wd: int,
                 dep_dev_wd: int,
                 dep_mean_we: int,
                 dep_dev_we: float,
                 min_dep: int,
                 max_dep: int
                 ):
        self.dep_mean_wd = dep_mean_wd
        self.dep_dev_wd = dep_dev_wd
        self.dep_mean_we = dep_mean_we
        self.dep_dev_we = dep_dev_we
        self.min_dep = min_dep
        self.max_dep = max_dep


class ReturnTime:
    def __init__(self,
                 ret_mean_wd: int,
                 ret_dev_wd: int,
                 ret_mean_we: int,
                 ret_dev_we: float,
                 min_return: int,
                 max_return: int
                 ):
        self.ret_mean_wd = ret_mean_wd
        self.ret_dev_wd = ret_dev_wd
        self.ret_mean_we = ret_mean_we
        self.ret_dev_we = ret_dev_we
        self.min_return = min_return
        self.max_return = max_return


class DistanceTravelled:
    def __init__(self,
                 avg_distance_wd: int,
                 dev_distance_wd: int,
                 avg_distance_we: int,
                 dev_distance_we: int,
                 min_distance: int,
                 max_distance: int
                 ):
        self.avg_distance_wd = avg_distance_wd
        self.dev_distance_wd = dev_distance_wd
        self.avg_distance_we = avg_distance_we
        self.dev_distance_we = dev_distance_we
        self.min_distance = min_distance
        self.max_distance = max_distance


class Consumption:
    def __init__(self,
                 consumption_mean: float,
                 consumption_std: float,
                 consumption_min: float,
                 consumption_max: float,
                 total_cons_clip: float
                 ):
        self.consumption_mean = consumption_mean
        self.consumption_std = consumption_std
        self.consumption_min = consumption_min
        self.consumption_max = consumption_max
        self.total_cons_clip = total_cons_clip


class Charger:
    def __init__(self,
                 charging_power
                 ):
        self.charging_power = charging_power


@registered_job
class ScheduleStatisticsJob(Job):

    def __init__(self,
                 departure_time: dict,
                 return_time: dict,
                 distance_travelled: dict,
                 consumption: dict,
                 charger: dict,
                 **kwargs):

        super().__init__(**kwargs)

        self.departure_time = DepartureTime(**departure_time)
        self.return_time = ReturnTime(**return_time)
        self.distance_travelled = DistanceTravelled(**distance_travelled)
        self.consumption = Consumption(**consumption)
        self.charger = Charger(**charger)

    @staticmethod
    def get_toml_key() -> str:
        return "schedule_statistics"

    def is_finished(self) -> bool:
        return True
