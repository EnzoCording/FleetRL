from enum import Enum

class ScheduleType(Enum):
    Delivery = 1
    Caretaker = 2
    Utility = 3

class ScheduleConfig:
    def __init__(self, schedule_type: ScheduleType):

        if schedule_type == schedule_type.Delivery:
            self.dep_mean_wd = 7  # mean departure time weekday
            self.dep_dev_wd = 1  # std deviation departure time weekday
            self.ret_mean_wd = 19  # mean return time weekday
            self.ret_dev_wd = 1  # std deviation return time weekday

            self.dep_mean_we = 9  # mean departure time weekend
            self.dep_dev_we = 1.5
            self.ret_mean_we = 17
            self.ret_dev_we = 1.5

            self.avg_distance_wd = 150  # mean distance travelled weekday
            self.dev_distance_wd = 25  # std deviation distance weekday
            self.avg_distance_we = 75  # mean distance weekend
            self.dev_distance_we = 25

        if schedule_type == schedule_type.Caretaker:
            self.dep_mean_wd = 7  # mean departure time weekday
            self.dep_dev_wd = 1  # std deviation departure time weekday
            self.ret_mean_wd = 19  # mean return time weekday
            self.ret_dev_wd = 1  # std deviation return time weekday

            self.dep_mean_we = 9  # mean departure time weekend
            self.dep_dev_we = 1.5
            self.ret_mean_we = 17
            self.ret_dev_we = 1.5

            self.avg_distance_wd = 150  # mean distance travelled weekday
            self.dev_distance_wd = 25  # std deviation distance weekday
            self.avg_distance_we = 75  # mean distance weekend
            self.dev_distance_we = 25

        if schedule_type == schedule_type.Utility:
            self.dep_mean_wd = 7  # mean departure time weekday
            self.dep_dev_wd = 1  # std deviation departure time weekday
            self.ret_mean_wd = 19  # mean return time weekday
            self.ret_dev_wd = 1  # std deviation return time weekday

            self.dep_mean_we = 9  # mean departure time weekend
            self.dep_dev_we = 1.5
            self.ret_mean_we = 17
            self.ret_dev_we = 1.5

            self.avg_distance_wd = 150  # mean distance travelled weekday
            self.dev_distance_wd = 25  # std deviation distance weekday
            self.avg_distance_we = 75  # mean distance weekend
            self.dev_distance_we = 25

        '''
        The following statistics are computed from the Emobpy German case and thus represent driving behaviour
        and consumption of the German population
        Source: https://zenodo.org/record/4514928
        '''

        self.consumption_mean = 0.221795291168322  # Average consumption in kWh/km
        self.consumption_std = 0.167463672468669  # Standard deviation of consumption in kWh/km
        self.consumption_min = 0.0994  # Minimum value of consumption, used as a floor for consumption levels
