from enum import Enum

class ScheduleType(Enum):
    Delivery = 1
    Caretaker = 2
    Utility = 3

class ScheduleConfig:
    def __init__(self, schedule_type: ScheduleType):

        if schedule_type == schedule_type.Delivery:
            # Benz eVito: https://www.mercedes-benz.de/vans/models/evito/panel-van/overview.html
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
            self.min_distance = 20
            self.max_distance = 280

            self.consumption_mean = 0.213  # Average consumption in kWh/km of Benz e-Vito
            self.consumption_std = 0.167463672468669  # Standard deviation of consumption in kWh/km
            self.consumption_min = 0.0994  # Minimum value of consumption, used as a floor for consumption levels
            self.consumption_max = 0.453  # Maximum consumption, ceiling of consumption levels

            self.min_return = 12  # Return hour must be bigger or equal to this value
            self.max_return_hour = 23  # Return hour must be smaller or equal to this value
            self.charging_power = 11  # Charging power in kW #TODO connect with company type

        if schedule_type == schedule_type.Caretaker:
            # Smart EQ: https://www.smart.mercedes-benz.com/de/de/modelle/smart-eq-fortwo-coupe
            self.dep_mean_wd = 6  # mean departure time weekday
            self.dep_dev_wd = 1  # std deviation departure time weekday
            self.pause_beg_mean_wd = 12  # mean pause beginning weekday
            self.pause_beg_dev_wd = 0.25  # std dev pause beginning weekday
            self.pause_end_mean_wd = 13.5  # mean pause end weekday
            self.pause_end_dev_wd = 0.25  # std dev pause end weekday
            self.ret_mean_wd = 19  # mean return time weekday
            self.ret_dev_wd = 1  # std deviation return time weekday
            self.min_return_wd = 15

            self.dep_mean_we = 9  # mean departure time weekend
            self.dep_dev_we = 1.5
            self.pause_beg_mean_we = 12  # mean pause beginning weekend
            self.pause_beg_dev_we = 0.25  # std dev pause beginning weekend
            self.pause_end_mean_we = 13  # mean pause end weekend
            self.pause_end_dev_we = 0.25  # std dev pause end weekend
            self.ret_mean_we = 15
            self.ret_dev_we = 1.5
            self.min_return_we = 15  # Return hour must be bigger or equal to this value
            self.max_return_hour = 23  # Return hour must be smaller or equal to this value

            self.prob_emergency = 0.02

            self.avg_distance_wd = 60  # mean distance travelled weekday
            self.dev_distance_wd = 10  # std deviation distance weekday
            self.avg_distance_we = 30  # mean distance weekend
            self.dev_distance_we = 20
            self.min_distance = 20
            self.max_distance = 80
            self.avg_distance_em = 15
            self.dev_distance_em = 15
            self.min_em_distance = 5

            self.consumption_mean = 0.17  # Average consumption in kWh/km of Smart EQ
            self.consumption_std = 0.167463672468669  # Standard deviation of consumption in kWh/km
            self.consumption_min = 0.0994  # Minimum value of consumption, used as a floor for consumption levels
            self.consumption_max = 0.453  # Maximum consumption, ceiling of consumption levels

            self.charging_power = 4.7  # kW

        if schedule_type == schedule_type.Utility:
            # Citroen e Berlingo: https://business.citroen.de/modellpalette/berlingo-kastenwagen.html
            self.dep_mean_wd = 7  # mean departure time weekday
            self.dep_dev_wd = 1  # std deviation departure time weekday
            self.ret_mean_wd = 19  # mean return time weekday
            self.ret_dev_wd = 1  # std deviation return time weekday

            self.dep_mean_we = 9  # mean departure time weekend
            self.dep_dev_we = 2
            self.ret_mean_we = 16
            self.ret_dev_we = 2
            self.max_return_hour = 23  # Return hour must be smaller or equal to this value
            self.min_return = 12  # Return hour must be bigger or equal to this value

            self.avg_distance_wd = 40  # mean distance travelled weekday
            self.dev_distance_wd = 30  # std deviation distance weekday
            self.avg_distance_we = 30  # mean distance weekend
            self.dev_distance_we = 25
            self.min_distance = 20
            self.max_distance = 220

            self.consumption_mean = 0.224  # Average consumption in kWh/km of e-Berlingo Citroen
            self.consumption_std = 0.167463672468669  # Standard deviation of consumption in kWh/km
            self.consumption_min = 0.0994  # Minimum value of consumption, used as a floor for consumption levels
            self.consumption_max = 0.453  # Maximum consumption, ceiling of consumption levels

            self.charging_power = 22  # kW

        '''
        The consumption std dev, min, max statistics are computed from the Emobpy German case and thus represent
        driving behaviour and consumption of the German population. Source: https://zenodo.org/record/4514928
        '''
