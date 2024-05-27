import datetime as dt
import math
from typing import Callable

import numpy as np
import pandas as pd

from fleetrl.utils.schedule.schedule_config import ScheduleAlgorithm
from fleetrl_2.jobs.schedule_statistics_job import Charger
from fleetrl_2.jobs.schedule_statistics_job import Consumption
from fleetrl_2.jobs.schedule_statistics_job import DepartureTime
from fleetrl_2.jobs.schedule_statistics_job import DistanceTravelled
from fleetrl_2.jobs.schedule_statistics_job import ReturnTime


class ScheduleGenerator:

    def __init__(self,
                 schedule_algorithm: Callable[["ScheduleGenerator", int], pd.DataFrame],
                 starting_date: str,
                 ending_date: str,
                 freq: str,
                 consumption: Consumption,
                 charger: Charger,
                 return_time: ReturnTime,
                 departure_time: DepartureTime,
                 distance_travelled: DistanceTravelled,
                 seed: int):

        """
        TODO
        """

        self.starting_date = starting_date
        self.ending_date = ending_date
        self.freq = freq
        self.schedule_algorithm = schedule_algorithm
        self.consumption = consumption
        self.charger = charger
        self.return_time = return_time
        self.departure_time = departure_time
        self.distance_travelled = distance_travelled
        self.seed = seed

        # define schedule type
        # self.sc = ScheduleConfig(schedule_type=self.schedule_algorithm,
        #                          env_config=env_config)

        # set starting, ending and frequency
        # self.starting_date = env_config["gen_start_date"]
        # self.ending_date = env_config["gen_end_date"]
        # self.freq = env_config["freq"]
        #
        # self.vehicle_id = vehicle_id

    def generate_schedule(self, ev_id: int):

        """
        This method chooses the right generation method depending on the use-case. Returns the schedule dataframe.

        :param ev_id: ID of the vehicle being scheduled
        :return: pd.DataFrame of the schedule
        """

        self.schedule_algorithm(self, ev_id)


    def generate_caretaker(self, ev_id: int):

        """
        Caretaker generator. Lunch break, operations on Sunday, chance for emergency trips at night
        :return: pd.DataFrame of the schedule
        """

        # make DataFrame and a date range, from start to end
        ev_schedule = pd.DataFrame()
        ev_schedule["date"] = pd.date_range(start=self.starting_date,
                                            end=self.ending_date,
                                            freq=self.freq)

        # Loop through each date entry and create the other entries
        for step in ev_schedule["date"]:

            # if new day, specify new random values
            if (step.hour == 0) and (step.minute == 0):

                # weekdays
                if step.weekday() < 5:

                    # time mean and std dev in config
                    dep_time = np.random.normal(self.sc.mean_wd,
                                                self.sc.dev_wd)
                    dep_hour = int(math.modf(dep_time)[1])
                    dep_hour = min([dep_hour, self.sc.max])
                    dep_hour = max([dep_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    pause_beg_time = np.random.normal(
                        self.sc.pause_beg_mean_wd, self.sc.pause_beg_dev_wd)
                    pause_beg_hour = int(math.modf(pause_beg_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(
                        math.modf(pause_beg_time)[0] * 60)).argmin()
                    pause_beg_min = minutes[closest_index]

                    pause_end_time = np.random.normal(
                        self.sc.pause_end_mean_wd, self.sc.pause_end_dev_wd)
                    pause_end_hour = int(math.modf(pause_end_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(
                        math.modf(pause_end_time)[0] * 60)).argmin()
                    pause_end_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.mean_wd,
                                                self.sc.dev_wd)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min_return_wd])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(ret_time)[0] * 60)).argmin()
                    ret_min = minutes[closest_index]

                    # make dates for easier comparison
                    dep_date = dt.datetime(step.year, step.month, step.day,
                                           hour=dep_hour, minute=dep_min)
                    pause_beg_date = dt.datetime(step.year, step.month,
                                                 step.day, hour=pause_beg_hour,
                                                 minute=pause_beg_min)
                    pause_end_date = dt.datetime(step.year, step.month,
                                                 step.day, hour=pause_end_hour,
                                                 minute=pause_end_min)
                    if (pause_end_date - pause_beg_date).total_seconds() < 0:
                        diff = (
                                pause_end_date - pause_beg_date).total_seconds()
                        pause_end_date += dt.timedelta(seconds=abs(diff))
                        pause_end_date += dt.timedelta(minutes=15)
                    ret_date = dt.datetime(step.year, step.month, step.day,
                                           hour=ret_hour, minute=ret_min)

                    # amount of time steps per trip
                    first_trip_steps = (
                                               pause_beg_date - dep_date).total_seconds() / 3600 * 4
                    second_trip_steps = (
                                                ret_date - pause_end_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.mean_wd,
                                                      self.sc.dev_wd)
                    total_distance = max(
                        [total_distance, self.sc.min])

                    # total distance travelled that day
                    total_distance = np.random.normal(self.sc.mean_wd,
                                                      self.sc.dev_wd)
                    total_distance = max(
                        [total_distance, self.sc.min])
                    total_distance = min(
                        [total_distance, self.sc.max])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # weekend
                else:
                    dep_time = np.random.normal(self.sc.mean_we,
                                                self.sc.dev_we)
                    dep_hour = int(math.modf(dep_time)[1])
                    dep_hour = min([dep_hour, self.sc.max])
                    dep_hour = max([dep_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    pause_beg_time = np.random.normal(
                        self.sc.pause_beg_mean_we, self.sc.pause_beg_dev_we)
                    pause_beg_hour = int(math.modf(pause_beg_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(
                        math.modf(pause_beg_time)[0] * 60)).argmin()
                    pause_beg_min = minutes[closest_index]

                    pause_end_time = np.random.normal(
                        self.sc.pause_end_mean_we, self.sc.pause_end_dev_we)
                    pause_end_hour = int(math.modf(pause_end_time)[1])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(minutes - int(
                        math.modf(pause_end_time)[0] * 60)).argmin()
                    pause_end_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.mean_we,
                                                self.sc.dev_we)
                    ret_hour = int(math.modf(ret_time)[1])
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min_return_we])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(ret_time)[0] * 60)).argmin()
                    ret_min = minutes[closest_index]

                    dep_date = dt.datetime(step.year, step.month, step.day,
                                           hour=dep_hour, minute=dep_min)
                    pause_beg_date = dt.datetime(step.year, step.month,
                                                 step.day, hour=pause_beg_hour,
                                                 minute=pause_beg_min)
                    pause_end_date = dt.datetime(step.year, step.month,
                                                 step.day, hour=pause_end_hour,
                                                 minute=pause_end_min)
                    if (pause_end_date - pause_beg_date).total_seconds() < 0:
                        diff = (
                                pause_end_date - pause_beg_date).total_seconds()
                        pause_end_date += dt.timedelta(seconds=abs(diff))
                        pause_end_date += dt.timedelta(minutes=15)
                    ret_date = dt.datetime(step.year, step.month, step.day,
                                           hour=ret_hour, minute=ret_min)

                    first_trip_steps = (
                                               pause_beg_date - dep_date).total_seconds() / 3600 * 4
                    second_trip_steps = (
                                                ret_date - pause_end_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.mean_we,
                                                      self.sc.dev_we)
                    total_distance = max(
                        [total_distance, self.sc.min])
                    total_distance = min(
                        [total_distance, self.sc.max])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

            # if trip is ongoing
            if (step >= dep_date) and (step < pause_beg_date):

                # dividing the total distance into equal parts
                ev_schedule.loc[ev_schedule[
                                    "date"] == step, "Distance_km"] = total_distance / first_trip_steps

                # sampling consumption in kWh / km based on Emobpy German case statistics
                # Clipping to min
                cons_rating = max([np.random.normal(self.sc.consumption_mean,
                                                    self.sc.consumption_std),
                                   self.sc.consumption_min])
                # Clipping to max
                cons_rating = min([cons_rating, self.sc.consumption_max])
                # Clipping such that the maximum amount of energy per trip is not exceeded
                cons_rating = min(
                    [cons_rating, self.sc.total_cons_clip / total_distance])
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Consumption_kWh"] = (
                                                                              total_distance / first_trip_steps) * cons_rating

                # set relevant entries
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Location"] = "driving"
                ev_schedule.loc[
                    ev_schedule["date"] == step, "ChargingStation"] = "none"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(
                    self.vehicle_id)
                ev_schedule.loc[
                    ev_schedule["date"] == step, "PowerRating_kW"] = 0.0

            elif (step >= pause_end_date) and (step < ret_date):
                # dividing the total distance into equal parts
                ev_schedule.loc[ev_schedule[
                                    "date"] == step, "Distance_km"] = total_distance / second_trip_steps

                # sampling consumption in kWh / km based on Emobpy German case statistics
                # Clipping to min
                cons_rating = max([np.random.normal(self.sc.consumption_mean,
                                                    self.sc.consumption_std),
                                   self.sc.consumption_min])
                # Clipping to max
                cons_rating = min([cons_rating, self.sc.consumption_max])
                # Clipping such that the maximum amount of energy per trip is not exceeded
                cons_rating = min([cons_rating,
                                   self.sc.total_cons_clip_afternoon / total_distance])
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Consumption_kWh"] = (
                                                                              total_distance / second_trip_steps) * cons_rating

                # set relevant entries
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Location"] = "driving"
                ev_schedule.loc[
                    ev_schedule["date"] == step, "ChargingStation"] = "none"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(
                    self.vehicle_id)
                ev_schedule.loc[
                    ev_schedule["date"] == step, "PowerRating_kW"] = 0.0

            else:
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Distance_km"] = 0.0
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Consumption_kWh"] = 0.0
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Location"] = "home"
                ev_schedule.loc[
                    ev_schedule["date"] == step, "ChargingStation"] = "home"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(
                    self.vehicle_id)
                ev_schedule.loc[ev_schedule[
                                    "date"] == step, "PowerRating_kW"] = self.sc.charging_power

            if step == dt.datetime(step.year, step.month, step.day, hour=23,
                                   minute=45):
                if np.random.random() > 0.98:
                    # emergency
                    em_start_date = dt.datetime(step.year, step.month,
                                                step.day, hour=2, minute=0)
                    em_end_date = dt.datetime(step.year, step.month, step.day,
                                              hour=4, minute=0)
                    dr = pd.date_range(start=em_start_date, end=em_end_date,
                                       freq="15T")
                    trip_steps = (
                                         em_end_date - em_start_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.avg_distance_em,
                                                      self.sc.dev_distance_em)
                    total_distance = max(
                        [total_distance, self.sc.min_em_distance])

                    for step in dr:
                        # dividing the total distance into equal parts
                        ev_schedule.loc[ev_schedule[
                                            "date"] == step, "Distance_km"] = total_distance / trip_steps

                        # sampling consumption in kWh / km based on Emobpy German case statistics
                        # Clipping to min
                        cons_rating = max([np.random.normal(
                            self.sc.consumption_mean, self.sc.consumption_std),
                            self.sc.consumption_min])
                        # Clipping to max
                        cons_rating = min(
                            [cons_rating, self.sc.consumption_max])
                        # Clipping such that the maximum amount of energy per trip is not exceeded
                        cons_rating = min([cons_rating,
                                           self.sc.total_cons_clip / total_distance])
                        ev_schedule.loc[ev_schedule[
                                            "date"] == step, "Consumption_kWh"] = (
                                                                                          total_distance / trip_steps) * cons_rating

                        # set relevant entries
                        ev_schedule.loc[ev_schedule[
                                            "date"] == step, "Location"] = "driving"
                        ev_schedule.loc[ev_schedule[
                                            "date"] == step, "ChargingStation"] = "none"
                        ev_schedule.loc[
                            ev_schedule["date"] == step, "ID"] = str(
                            self.vehicle_id)
                        ev_schedule.loc[ev_schedule[
                                            "date"] == step, "PowerRating_kW"] = 0.0

        return ev_schedule

    def generate_utility(self, ev_id: int):

        """
        Utility generation. Chance for operations on Sunday.
        :return: pd.DataFrame of the schedule.
        """

        # make DataFrame and a date range, from start to end
        ev_schedule = pd.DataFrame()
        ev_schedule["date"] = pd.date_range(start=self.starting_date,
                                            end=self.ending_date,
                                            freq=self.freq)

        if ev_schedule["date"][0].weekday() == 6:
            print("First day is a Sunday, skipping it...")

            while ev_schedule["date"][0].weekday() == 6:
                ev_schedule.drop(index=0, inplace=True)

            assert ev_schedule["date"][
                       0].weekday != 6, "Error, first day is still a Sunday."

            new_start = ev_schedule["date"][0]
            print(f"Now starting on date: {new_start}")

        # Loop through each date entry and create the other entries
        for step in ev_schedule["date"]:

            # if new day, specify new random values
            if (step.hour == 0) and (step.minute == 0):

                # weekdays
                if step.weekday() < 5:

                    # time mean and std dev in config
                    dep_time = np.random.normal(self.sc.mean_wd,
                                                self.sc.dev_wd)
                    # split number and decimals, use number and turn to int
                    dep_hour = int(math.modf(dep_time)[1])
                    dep_hour = min([dep_hour, self.sc.max])
                    dep_hour = max([dep_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    # split number and decimals, use decimals and choose the closest minute
                    closest_index = np.abs(
                        minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.mean_wd,
                                                self.sc.dev_wd)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(ret_time)[0] * 60)).argmin()
                    ret_min = minutes[closest_index]

                    # make dates for easier comparison
                    dep_date = dt.datetime(step.year, step.month, step.day,
                                           hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day,
                                           hour=ret_hour, minute=ret_min)

                    # amount of time steps per trip
                    trip_steps = (
                                         ret_date - dep_date).total_seconds() / 3600 * 4

                    # total distance travelled that day
                    total_distance = np.random.normal(self.sc.mean_wd,
                                                      self.sc.dev_wd)
                    total_distance = max(
                        [total_distance, self.sc.min])
                    total_distance = min(
                        [total_distance, self.sc.max])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # weekend
                elif step.weekday() == 5:
                    dep_time = np.random.normal(self.sc.mean_we,
                                                self.sc.dev_we)
                    dep_hour = int(math.modf(dep_time)[1])
                    dep_hour = min([dep_hour, self.sc.max])
                    dep_hour = max([dep_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.mean_we,
                                                self.sc.dev_we)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(ret_time)[0] * 60)).argmin()
                    ret_min = minutes[closest_index]

                    dep_date = dt.datetime(step.year, step.month, step.day,
                                           hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day,
                                           hour=ret_hour, minute=ret_min)

                    if dep_date > ret_date:
                        raise RuntimeError(
                            "Schedule statistics produce unrealistic schedule. dep > ret.")

                    trip_steps = (
                                         ret_date - dep_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.mean_we,
                                                      self.sc.dev_we)
                    total_distance = max(
                        [total_distance, self.sc.min])
                    total_distance = min(
                        [total_distance, self.sc.max])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                elif (step.weekday() == 6) and (np.random.random() > 0.95):
                    dep_time = np.random.normal(self.sc.mean_we,
                                                self.sc.dev_we)
                    dep_hour = int(math.modf(dep_time)[1])
                    dep_hour = min([dep_hour, self.sc.max])
                    dep_hour = max([dep_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.mean_we,
                                                self.sc.dev_we)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(ret_time)[0] * 60)).argmin()
                    ret_min = minutes[closest_index]

                    dep_date = dt.datetime(step.year, step.month, step.day,
                                           hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day,
                                           hour=ret_hour, minute=ret_min)

                    if dep_date > ret_date:
                        raise RuntimeError(
                            "Schedule statistics produce unrealistic schedule. dep > ret.")

                    trip_steps = (
                                         ret_date - dep_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.mean_we,
                                                      self.sc.dev_we)
                    total_distance = max(
                        [total_distance, self.sc.min])
                    total_distance = min(
                        [total_distance, self.sc.max])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # Assuming normally no operation on Sundays, apart from unlikely emergencies

            # if trip is ongoing
            if (step >= dep_date) and (step < ret_date):

                # dividing the total distance into equal parts
                ev_schedule.loc[ev_schedule[
                                    "date"] == step, "Distance_km"] = total_distance / trip_steps

                # sampling consumption in kWh / km based on Emobpy German case statistics
                # Clipping to min
                cons_rating = max([np.random.normal(self.sc.consumption_mean,
                                                    self.sc.consumption_std),
                                   self.sc.consumption_min])
                # Clipping to max
                cons_rating = min([cons_rating, self.sc.consumption_max])
                # Clipping such that the maximum amount of energy per trip is not exceeded
                cons_rating = min(
                    [cons_rating, self.sc.total_cons_clip / total_distance])
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Consumption_kWh"] = (
                                                                              total_distance / trip_steps) * cons_rating

                # set relevant entries
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Location"] = "driving"
                ev_schedule.loc[
                    ev_schedule["date"] == step, "ChargingStation"] = "none"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(
                    self.vehicle_id)
                ev_schedule.loc[
                    ev_schedule["date"] == step, "PowerRating_kW"] = 0.0

            else:
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Distance_km"] = 0.0
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Consumption_kWh"] = 0.0
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Location"] = "home"
                ev_schedule.loc[
                    ev_schedule["date"] == step, "ChargingStation"] = "home"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(
                    self.vehicle_id)
                ev_schedule.loc[ev_schedule[
                                    "date"] == step, "PowerRating_kW"] = self.sc.charging_power

        return ev_schedule

    def generate_custom(self, ev_id: int):

        """
        Custom schedule generator. Saturdays operations occur but at reduced levels, no operations on Sunday.
        :return: pd.DataFrame of the schedule
        """

        # make DataFrame and a date range, from start to end
        ev_schedule = pd.DataFrame()
        ev_schedule["date"] = pd.date_range(start=self.starting_date,
                                            end=self.ending_date,
                                            freq=self.freq)

        if ev_schedule["date"][0].weekday() == 6:
            print("First day is a Sunday, skipping it...")

            while ev_schedule["date"][0].weekday() == 6:
                ev_schedule.drop(index=0, inplace=True)

            assert ev_schedule["date"][
                       0].weekday != 6, "Error, first day is still a Sunday."

            new_start = ev_schedule["date"][0]
            print(f"Now starting on date: {new_start}")

        # Loop through each date entry and create the other entries
        for step in ev_schedule["date"]:

            # if new day, specify new random values
            if (step.hour == 0) and (step.minute == 0):

                # weekdays
                if step.weekday() < 5:

                    # time mean and std dev in config
                    dep_time = np.random.normal(self.sc.mean_wd,
                                                self.sc.dev_wd)
                    # split number and decimals, use number and turn to int
                    dep_hour = int(math.modf(dep_time)[1])
                    dep_hour = min([dep_hour, self.sc.max])
                    dep_hour = max([dep_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    # split number and decimals, use decimals and choose the closest minute
                    closest_index = np.abs(
                        minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.mean_wd,
                                                self.sc.dev_wd)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(ret_time)[0] * 60)).argmin()
                    ret_min = minutes[closest_index]

                    # make dates for easier comparison
                    dep_date = dt.datetime(step.year, step.month, step.day,
                                           hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day,
                                           hour=ret_hour, minute=ret_min)

                    # amount of time steps per trip
                    trip_steps = (
                                         ret_date - dep_date).total_seconds() / 3600 * 4

                    # total distance travelled that day
                    total_distance = np.random.normal(self.sc.mean_wd,
                                                      self.sc.dev_wd)
                    total_distance = max(
                        [total_distance, self.sc.min])
                    total_distance = min(
                        [total_distance, self.sc.max])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # weekend
                elif step.weekday() == 5:
                    dep_time = np.random.normal(self.sc.mean_we,
                                                self.sc.dev_we)
                    dep_hour = int(math.modf(dep_time)[1])
                    dep_hour = min([dep_hour, self.sc.max])
                    dep_hour = max([dep_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.sc.mean_we,
                                                self.sc.dev_we)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.sc.max_return_hour])
                    ret_hour = max([ret_hour, self.sc.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(ret_time)[0] * 60)).argmin()
                    ret_min = minutes[closest_index]

                    dep_date = dt.datetime(step.year, step.month, step.day,
                                           hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day,
                                           hour=ret_hour, minute=ret_min)

                    if dep_date > ret_date:
                        raise RuntimeError(
                            "Schedule statistics produce unrealistic schedule. dep > ret.")

                    trip_steps = (
                                         ret_date - dep_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.mean_we,
                                                      self.sc.dev_we)
                    total_distance = max(
                        [total_distance, self.sc.min])
                    total_distance = min(
                        [total_distance, self.sc.max])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # Assuming no operation on Sundays

            # if trip is ongoing
            if (step >= dep_date) and (step < ret_date):

                # dividing the total distance into equal parts
                ev_schedule.loc[ev_schedule[
                                    "date"] == step, "Distance_km"] = total_distance / trip_steps

                # sampling consumption in kWh / km based on Emobpy German case statistics
                # Clipping to min
                cons_rating = max([np.random.normal(self.sc.consumption_mean,
                                                    self.sc.consumption_std),
                                   self.sc.consumption_min])
                # Clipping to max
                cons_rating = min([cons_rating, self.sc.consumption_max])
                # Clipping such that the maximum amount of energy per trip is not exceeded
                cons_rating = min(
                    [cons_rating, self.sc.total_cons_clip / total_distance])

                ev_schedule.loc[
                    ev_schedule["date"] == step, "Consumption_kWh"] = (
                                                                              total_distance / trip_steps) * cons_rating

                # set relevant entries
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Location"] = "driving"
                ev_schedule.loc[
                    ev_schedule["date"] == step, "ChargingStation"] = "none"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(
                    self.vehicle_id)
                ev_schedule.loc[
                    ev_schedule["date"] == step, "PowerRating_kW"] = 0.0

            else:
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Distance_km"] = 0.0
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Consumption_kWh"] = 0.0
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Location"] = "home"
                ev_schedule.loc[
                    ev_schedule["date"] == step, "ChargingStation"] = "home"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(
                    self.vehicle_id)
                ev_schedule.loc[ev_schedule[
                                    "date"] == step, "PowerRating_kW"] = self.sc.charging_power

        return ev_schedule
    #
    # _schedule_algorithm = {
    #     ScheduleAlgorithm.Delivery: generate_delivery,
    #     ScheduleAlgorithm.Utility: generate_utility,
    #     ScheduleAlgorithm.Caretaker: generate_caretaker,
    # }
    #
    # def get_schedule_algorithm(self):
    #     a = ScheduleGenerator._schedule_algorithm[self.schedule_algorithm]
    #     return lambda ev_id: a(self, ev_id)
    #
    # @staticmethod
    # def add_schedule_algorithm(algorithm_name: str,
    #         schedule_algorithm: Callable[["ScheduleGenerator", int], pd.DataFrame]):
    #     ScheduleGenerator._schedule_algorithm[algorithm_name] = schedule_algorithm

if __name__ == '__main__':
    sg = ScheduleGenerator.__new__(ScheduleGenerator.__class__)
    sg.alg
    sg.get_schedule_algorithm()
