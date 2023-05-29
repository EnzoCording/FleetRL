from enum import Enum

class ScheduleType(Enum):
    Delivery = 1
    Caretaker = 2
    Utility = 3

class ScheduleConfig:
    def __init__(self, schedule_type: ScheduleType):
        if schedule_type == schedule_type.Delivery:
            self.dep_mean_wd = 7
            self.dep_dev_wd = 1
            self.ret_mean_wd = 19
            self.ret_dev_wd = 1

            self.dep_mean_we = 9
            self.dep_dev_we = 1.5
            self.ret_mean_we = 17
            self.ret_dev_we = 1.5

            self.avg_distance_wd = 150
            self.dev_distance_wd = 25
            self.avg_distance_we = 75
            self.dev_distance_we = 25

        self.consumption_mean = 0.221795291168322
        self.consumption_std = 0.167463672468669
        self.consumption_min = 0.0994
