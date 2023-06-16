import datetime
import math
import pandas as pd
import numpy as np
import datetime as dt
import time
import os

from FleetRL.utils.schedule_generator.schedule_config import ScheduleConfig, ScheduleType

class ScheduleGenerator:

    def __init__(self, schedule_dir: str, starting_date: str = "01/01/2020",
                 ending_date: str = "30/12/2020",
                 freq: str = "15T",
                 save_schedule: bool = True,
                 file_comment: str = "",
                 schedule_type: ScheduleType = ScheduleType.Delivery,
                 vehicle_id: str = "0"):

        # Set seed for reproducibility
        np.random.seed(42)

        # define schedule type
        self.schedule_type = schedule_type
        self.sc = ScheduleConfig(self.schedule_type)

        # set starting, ending and frequency
        self.starting_date = starting_date
        self.ending_date = ending_date
        self.freq = freq

        self.vehicle_id = vehicle_id

        # flag whether to save the schedule as a csv or not
        self.save = save_schedule

        # get time to make unique file names
        self.time_now = int(time.time())

        # same dir as the other schedules
        self.file_comment = file_comment
        self.schedule_dir = schedule_dir
        self.file_name = f"schedule_{self.time_now}_{self.file_comment}.csv"
        self.path_name = self.schedule_dir + self.file_name

        # make dir if not existing
        if not os.path.exists(self.schedule_dir):
            os.makedirs(self.schedule_dir)

    def get_file_name(self):
        return self.file_name

    def generate_schedule(self):
        if self.schedule_type == self.schedule_type.Delivery:
            return self.generate_delivery()
        elif self.schedule_type == self.schedule_type.Caretaker:
            return self.generate_caretaker()
        elif self.schedule_type == self.schedule_type.Utility:
            return self.generate_utility()
        else:
            raise TypeError("Company type not found!")

    def generate_delivery(self):

        # make DataFrame and a date range, from start to end
        ev_schedule = pd.DataFrame()
        ev_schedule["date"] = pd.date_range(start=self.starting_date, end=self.ending_date, freq = self.freq)

        # Loop through each date entry and create the other entries
        for step in ev_schedule["date"]:

            # if new day, specify new random values
            if (step.hour == 0) and (step.minute == 0):

                # weekdays
                if step.weekday() < 5:

                    # time mean and std dev in config
                    dep_time = np.random.normal(self.sc.dep_mean_wd, self.sc.dep_dev_wd)
                    # split number and decimals, use number and turn to int
                    dep_hour = int(math.modf(dep_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    # split number and decimals, use decimals and choose the closest minute
                    closest_index = np.abs(minutes - int(math.modf(dep_time)[0]*60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.ret_mean_wd, self.sc.ret_dev_wd)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min_return])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(ret_time)[0]*60)).argmin()
                    ret_min = minutes[closest_index]

                    # make dates for easier comparison
                    dep_date = dt.datetime(step.year, step.month, step.day, hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day, hour=ret_hour, minute=ret_min)

                    # amount of time steps per trip
                    trip_steps = (ret_date - dep_date).total_seconds() / 3600 * 4

                    # total distance travelled that day
                    total_distance = np.random.normal(self.sc.avg_distance_wd, self.sc.dev_distance_wd)
                    total_distance = max([total_distance, self.sc.min_distance])
                    total_distance = min([total_distance, self.sc.max_distance])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # weekend
                elif step.weekday() == 5:
                    dep_time = np.random.normal(self.sc.dep_mean_we, self.sc.dep_dev_we)
                    dep_hour = int(math.modf(dep_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(dep_time)[0]*60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.ret_mean_we, self.sc.ret_dev_we)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min_return])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(ret_time)[0]*60)).argmin()
                    ret_min = minutes[closest_index]

                    dep_date = dt.datetime(step.year, step.month, step.day, hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day, hour=ret_hour, minute=ret_min)

                    if dep_date > ret_date:
                        raise RuntimeError("Schedule statistics produce unrealistic schedule. dep > ret.")

                    trip_steps = (ret_date - dep_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.avg_distance_we, self.sc.dev_distance_we)
                    total_distance = max([total_distance, self.sc.min_distance])
                    total_distance = min([total_distance, self.sc.max_distance])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # Assuming no operation on Sundays

            # if trip is ongoing
            if (step >= dep_date) and (step < ret_date):

                # dividing the total distance into equal parts
                ev_schedule.loc[ev_schedule["date"] == step, "Distance_km"] = total_distance / trip_steps

                # sampling consumption in kWh / km based on Emobpy German case statistics
                # Clipping to min
                cons_rating = max([np.random.normal(self.sc.consumption_mean, self.sc.consumption_std),
                                   self.sc.consumption_min])
                # Clipping to max
                cons_rating = min([cons_rating, self.sc.consumption_max])
                ev_schedule.loc[ev_schedule["date"] == step, "Consumption_kWh"] = (total_distance / trip_steps) * cons_rating

                # set relevant entries
                ev_schedule.loc[ev_schedule["date"] == step, "Location"] = "driving"
                ev_schedule.loc[ev_schedule["date"] == step, "ChargingStation"] = "none"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(self.vehicle_id)
                ev_schedule.loc[ev_schedule["date"] == step, "PowerRating_kW"] = 0.0

            else:
                ev_schedule.loc[ev_schedule["date"] == step, "Distance_km"] = 0.0
                ev_schedule.loc[ev_schedule["date"] == step, "Consumption_kWh"] = 0.0
                ev_schedule.loc[ev_schedule["date"] == step, "Location"] = "home"
                ev_schedule.loc[ev_schedule["date"] == step, "ChargingStation"] = "home"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(self.vehicle_id)
                ev_schedule.loc[ev_schedule["date"] == step, "PowerRating_kW"] = self.sc.charging_power

        if self.save:
            ev_schedule.to_csv(self.path_name)

        return ev_schedule

    def generate_caretaker(self):

        # make DataFrame and a date range, from start to end
        ev_schedule = pd.DataFrame()
        ev_schedule["date"] = pd.date_range(start=self.starting_date, end=self.ending_date, freq = self.freq)

        # Loop through each date entry and create the other entries
        for step in ev_schedule["date"]:

            # if new day, specify new random values
            if (step.hour == 0) and (step.minute == 0):

                # weekdays
                if step.weekday() < 5:

                    # time mean and std dev in config
                    dep_time = np.random.normal(self.sc.dep_mean_wd, self.sc.dep_dev_wd)
                    dep_hour = int(math.modf(dep_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(dep_time)[0]*60)).argmin()
                    dep_min = minutes[closest_index]

                    pause_beg_time = np.random.normal(self.sc.pause_beg_mean_wd, self.sc.pause_beg_dev_wd)
                    pause_beg_hour = int(math.modf(pause_beg_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(pause_beg_time)[0] * 60)).argmin()
                    pause_beg_min = minutes[closest_index]

                    pause_end_time = np.random.normal(self.sc.pause_end_mean_wd, self.sc.pause_end_dev_wd)
                    pause_end_hour = int(math.modf(pause_end_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(pause_end_time)[0] * 60)).argmin()
                    pause_end_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.ret_mean_wd, self.sc.ret_dev_wd)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min_return_wd])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(ret_time)[0] * 60)).argmin()
                    ret_min = minutes[closest_index]

                    # make dates for easier comparison
                    dep_date = dt.datetime(step.year, step.month, step.day, hour=dep_hour, minute=dep_min)
                    pause_beg_date = dt.datetime(step.year, step.month, step.day, hour=pause_beg_hour, minute=pause_beg_min)
                    pause_end_date = dt.datetime(step.year, step.month, step.day, hour=pause_end_hour, minute=pause_end_min)
                    if (pause_end_date - pause_beg_date).total_seconds() < 0:
                        diff = (pause_end_date - pause_beg_date).total_seconds()
                        pause_end_date += dt.timedelta(seconds=abs(diff))
                        pause_end_date += dt.timedelta(minutes=15)
                    ret_date = dt.datetime(step.year, step.month, step.day, hour=ret_hour, minute=ret_min)

                    # amount of time steps per trip
                    first_trip_steps = (pause_beg_date - dep_date).total_seconds() / 3600 * 4
                    second_trip_steps = (ret_date - pause_end_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.avg_distance_wd, self.sc.dev_distance_wd)
                    total_distance = max([total_distance, self.sc.min_distance])

                    # total distance travelled that day
                    total_distance = np.random.normal(self.sc.avg_distance_wd, self.sc.dev_distance_wd)
                    total_distance = max([total_distance, self.sc.min_distance])
                    total_distance = min([total_distance, self.sc.max_distance])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # weekend
                else:
                    dep_time = np.random.normal(self.sc.dep_mean_we, self.sc.dep_dev_we)
                    dep_hour = int(math.modf(dep_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    pause_beg_time = np.random.normal(self.sc.pause_beg_mean_we, self.sc.pause_beg_dev_we)
                    pause_beg_hour = int(math.modf(pause_beg_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(pause_beg_time)[0] * 60)).argmin()
                    pause_beg_min = minutes[closest_index]

                    pause_end_time = np.random.normal(self.sc.pause_end_mean_we, self.sc.pause_end_dev_we)
                    pause_end_hour = int(math.modf(pause_end_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(pause_end_time)[0] * 60)).argmin()
                    pause_end_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.ret_mean_we, self.sc.ret_dev_we)
                    ret_hour = int(math.modf(ret_time)[1])
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min_return_we])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(ret_time)[0] * 60)).argmin()
                    ret_min = minutes[closest_index]

                    dep_date = dt.datetime(step.year, step.month, step.day, hour=dep_hour, minute=dep_min)
                    pause_beg_date = dt.datetime(step.year, step.month, step.day, hour=pause_beg_hour, minute=pause_beg_min)
                    pause_end_date = dt.datetime(step.year, step.month, step.day, hour=pause_end_hour, minute=pause_end_min)
                    if (pause_end_date - pause_beg_date).total_seconds() < 0:
                        diff = (pause_end_date - pause_beg_date).total_seconds()
                        pause_end_date += dt.timedelta(seconds=abs(diff))
                        pause_end_date += dt.timedelta(minutes=15)
                    ret_date = dt.datetime(step.year, step.month, step.day, hour=ret_hour, minute=ret_min)

                    first_trip_steps = (pause_beg_date - dep_date).total_seconds() / 3600 * 4
                    second_trip_steps = (ret_date - pause_end_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.avg_distance_we, self.sc.dev_distance_we)
                    total_distance = max([total_distance, self.sc.min_distance])
                    total_distance = min([total_distance, self.sc.max_distance])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

            # if trip is ongoing
            if (step >= dep_date) and (step < pause_beg_date):

                # dividing the total distance into equal parts
                ev_schedule.loc[ev_schedule["date"] == step, "Distance_km"] = total_distance / first_trip_steps

                # sampling consumption in kWh / km based on Emobpy German case statistics
                # Clipping to min
                cons_rating = max([np.random.normal(self.sc.consumption_mean, self.sc.consumption_std),
                                   self.sc.consumption_min])
                # Clipping to max
                cons_rating = min([cons_rating, self.sc.consumption_max])
                ev_schedule.loc[ev_schedule["date"] == step, "Consumption_kWh"] = (total_distance / first_trip_steps) * cons_rating

                # set relevant entries
                ev_schedule.loc[ev_schedule["date"] == step, "Location"] = "driving"
                ev_schedule.loc[ev_schedule["date"] == step, "ChargingStation"] = "none"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(self.vehicle_id)
                ev_schedule.loc[ev_schedule["date"] == step, "PowerRating_kW"] = 0.0

            elif (step >= pause_end_date) and (step < ret_date):
                # dividing the total distance into equal parts
                ev_schedule.loc[ev_schedule["date"] == step, "Distance_km"] = total_distance / second_trip_steps

                # sampling consumption in kWh / km based on Emobpy German case statistics
                # Clipping to min
                cons_rating = max([np.random.normal(self.sc.consumption_mean, self.sc.consumption_std),
                                   self.sc.consumption_min])
                # Clipping to max
                cons_rating = min([cons_rating, self.sc.consumption_max])
                ev_schedule.loc[ev_schedule["date"] == step, "Consumption_kWh"] = (total_distance / second_trip_steps) * cons_rating

                # set relevant entries
                ev_schedule.loc[ev_schedule["date"] == step, "Location"] = "driving"
                ev_schedule.loc[ev_schedule["date"] == step, "ChargingStation"] = "none"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(self.vehicle_id)
                ev_schedule.loc[ev_schedule["date"] == step, "PowerRating_kW"] = 0.0

            else:
                ev_schedule.loc[ev_schedule["date"] == step, "Distance_km"] = 0.0
                ev_schedule.loc[ev_schedule["date"] == step, "Consumption_kWh"] = 0.0
                ev_schedule.loc[ev_schedule["date"] == step, "Location"] = "home"
                ev_schedule.loc[ev_schedule["date"] == step, "ChargingStation"] = "home"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(self.vehicle_id)
                ev_schedule.loc[ev_schedule["date"] == step, "PowerRating_kW"] = self.sc.charging_power

            if step == dt.datetime(step.year, step.month, step.day, hour=23, minute=45):
                if np.random.random() > 0.98:
                    # emergency
                    em_start_date = dt.datetime(step.year, step.month, step.day, hour=2, minute=0)
                    em_end_date = dt.datetime(step.year, step.month, step.day, hour=4, minute=0)
                    dr = pd.date_range(start=em_start_date, end=em_end_date, freq="15T")
                    trip_steps = (em_end_date - em_start_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.avg_distance_em, self.sc.dev_distance_em)
                    total_distance = max([total_distance, self.sc.min_em_distance])

                    for step in dr:
                        # dividing the total distance into equal parts
                        ev_schedule.loc[ev_schedule["date"] == step, "Distance_km"] = total_distance / trip_steps

                        # sampling consumption in kWh / km based on Emobpy German case statistics
                        # Clipping to min
                        cons_rating = max([np.random.normal(self.sc.consumption_mean, self.sc.consumption_std),
                                           self.sc.consumption_min])
                        # Clipping to max
                        cons_rating = min([cons_rating, self.sc.consumption_max])
                        ev_schedule.loc[ev_schedule[
                                            "date"] == step, "Consumption_kWh"] = (total_distance / trip_steps) * cons_rating

                        # set relevant entries
                        ev_schedule.loc[ev_schedule["date"] == step, "Location"] = "driving"
                        ev_schedule.loc[ev_schedule["date"] == step, "ChargingStation"] = "none"
                        ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(self.vehicle_id)
                        ev_schedule.loc[ev_schedule["date"] == step, "PowerRating_kW"] = 0.0

        if self.save:
            ev_schedule.to_csv(self.path_name)

        return ev_schedule

    def generate_utility(self):

        # make DataFrame and a date range, from start to end
        ev_schedule = pd.DataFrame()
        ev_schedule["date"] = pd.date_range(start=self.starting_date, end=self.ending_date, freq = self.freq)

        # Loop through each date entry and create the other entries
        for step in ev_schedule["date"]:

            # if new day, specify new random values
            if (step.hour == 0) and (step.minute == 0):

                # weekdays
                if step.weekday() < 5:

                    # time mean and std dev in config
                    dep_time = np.random.normal(self.sc.dep_mean_wd, self.sc.dep_dev_wd)
                    # split number and decimals, use number and turn to int
                    dep_hour = int(math.modf(dep_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    # split number and decimals, use decimals and choose the closest minute
                    closest_index = np.abs(minutes - int(math.modf(dep_time)[0]*60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.ret_mean_wd, self.sc.ret_dev_wd)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min_return])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(ret_time)[0]*60)).argmin()
                    ret_min = minutes[closest_index]

                    # make dates for easier comparison
                    dep_date = dt.datetime(step.year, step.month, step.day, hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day, hour=ret_hour, minute=ret_min)

                    # amount of time steps per trip
                    trip_steps = (ret_date - dep_date).total_seconds() / 3600 * 4

                    # total distance travelled that day
                    total_distance = np.random.normal(self.sc.avg_distance_wd, self.sc.dev_distance_wd)
                    total_distance = max([total_distance, self.sc.min_distance])
                    total_distance = min([total_distance, self.sc.max_distance])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # weekend
                elif step.weekday() == 5:
                    dep_time = np.random.normal(self.sc.dep_mean_we, self.sc.dep_dev_we)
                    dep_hour = int(math.modf(dep_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(dep_time)[0]*60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.ret_mean_we, self.sc.ret_dev_we)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min_return])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(ret_time)[0]*60)).argmin()
                    ret_min = minutes[closest_index]

                    dep_date = dt.datetime(step.year, step.month, step.day, hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day, hour=ret_hour, minute=ret_min)

                    if dep_date > ret_date:
                        raise RuntimeError("Schedule statistics produce unrealistic schedule. dep > ret.")

                    trip_steps = (ret_date - dep_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.avg_distance_we, self.sc.dev_distance_we)
                    total_distance = max([total_distance, self.sc.min_distance])
                    total_distance = min([total_distance, self.sc.max_distance])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                elif (step.weekday() == 6) and (np.random.random() > 0.95):
                    dep_time = np.random.normal(self.sc.dep_mean_we, self.sc.dep_dev_we)
                    dep_hour = int(math.modf(dep_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.ret_mean_we, self.sc.ret_dev_we)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min_return])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(math.modf(ret_time)[0] * 60)).argmin()
                    ret_min = minutes[closest_index]

                    dep_date = dt.datetime(step.year, step.month, step.day, hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day, hour=ret_hour, minute=ret_min)

                    if dep_date > ret_date:
                        raise RuntimeError("Schedule statistics produce unrealistic schedule. dep > ret.")

                    trip_steps = (ret_date - dep_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.avg_distance_we, self.sc.dev_distance_we)
                    total_distance = max([total_distance, self.sc.min_distance])
                    total_distance = min([total_distance, self.sc.max_distance])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # Assuming normally no operation on Sundays, apart from unlikely emergencies

            # if trip is ongoing
            if (step >= dep_date) and (step < ret_date):

                # dividing the total distance into equal parts
                ev_schedule.loc[ev_schedule["date"] == step, "Distance_km"] = total_distance / trip_steps

                # sampling consumption in kWh / km based on Emobpy German case statistics
                # Clipping to min
                cons_rating = max([np.random.normal(self.sc.consumption_mean, self.sc.consumption_std),
                                   self.sc.consumption_min])
                # Clipping to max
                cons_rating = min([cons_rating, self.sc.consumption_max])
                ev_schedule.loc[ev_schedule["date"] == step, "Consumption_kWh"] = (total_distance / trip_steps) * cons_rating

                # set relevant entries
                ev_schedule.loc[ev_schedule["date"] == step, "Location"] = "driving"
                ev_schedule.loc[ev_schedule["date"] == step, "ChargingStation"] = "none"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(self.vehicle_id)
                ev_schedule.loc[ev_schedule["date"] == step, "PowerRating_kW"] = 0.0

            else:
                ev_schedule.loc[ev_schedule["date"] == step, "Distance_km"] = 0.0
                ev_schedule.loc[ev_schedule["date"] == step, "Consumption_kWh"] = 0.0
                ev_schedule.loc[ev_schedule["date"] == step, "Location"] = "home"
                ev_schedule.loc[ev_schedule["date"] == step, "ChargingStation"] = "home"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(self.vehicle_id)
                ev_schedule.loc[ev_schedule["date"] == step, "PowerRating_kW"] = self.sc.charging_power

        if self.save:
            ev_schedule.to_csv(self.path_name)

        return ev_schedule

    # def generate_multiple_ev_schedule(self, num_evs):
    #     schedule = pd.DataFrame()
    #     self.save = False
    #     self.file_comment += f"_{num_evs}_evs"
    #     self.file_name = f"schedule_{self.time_now}_{self.file_comment}.csv"
    #     self.path_name = self.schedule_dir + self.file_name
    #     for i in range(num_evs):
    #         self.vehicle_id = i
    #         new_schedule = self.generate_schedule()
    #         schedule = pd.concat([schedule, new_schedule], axis = 0, ignore_index=True)
    #     schedule.to_csv(self.path_name)
