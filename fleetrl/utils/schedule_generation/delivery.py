import datetime as dt
import logging
import math

import numpy as np
import pandas as pd
from tomlchef.package_manager import PackageManager

from fleetrl.utils.schedule_generation.schedule_generator import \
    ScheduleGenerator

logger = logging.getLogger(PackageManager.get_name())


class DeliveryScheduleGenerator(ScheduleGenerator):
    def _generate(self, ev_id: int) -> pd.DataFrame:
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
                    dep_time = np.random.normal(self.departure_time.mean_wd,
                                                self.departure_time.dev_wd)
                    # split number and decimals, use number and turn to int
                    dep_hour = int(math.modf(dep_time)[1])
                    dep_hour = min([dep_hour, self.departure_time.max])
                    dep_hour = max([dep_hour, self.departure_time.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    # split number and decimals, use decimals and choose the closest minute
                    closest_index = np.abs(
                        minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.return_time.mean_wd,
                                                self.return_time.dev_wd)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min([ret_hour, self.return_time.max])
                    ret_hour = max([ret_hour, self.return_time.min])
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
                    trip_steps = ((ret_date - dep_date).total_seconds()
                                  / 3600 * 4)

                    # total distance travelled that day
                    total_distance = np.random.normal(
                        self.distance_travelled.mean_wd,
                        self.distance_travelled.dev_wd)
                    total_distance = max(
                        [total_distance, self.distance_travelled.min])
                    total_distance = min(
                        [total_distance, self.distance_travelled.max])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # weekend
                elif step.weekday() == 5:
                    dep_time = np.random.normal(self.departure_time.mean_we,
                                                self.departure_time.dev_we)
                    dep_hour = int(math.modf(dep_time)[1])
                    dep_hour = min([dep_hour, self.departure_time.max])
                    dep_hour = max([dep_hour, self.departure_time.min])
                    minutes = np.asarray([0, 15, 30, 45])
                    closest_index = np.abs(
                        minutes - int(math.modf(dep_time)[0] * 60)).argmin()
                    dep_min = minutes[closest_index]

                    ret_time = np.random.normal(self.return_time.mean_we,
                                                self.return_time.dev_we)
                    ret_hour = int(math.modf(ret_time)[1])
                    # clip return to a maximum
                    ret_hour = min(
                        [ret_hour, self.return_time.max])
                    ret_hour = max([ret_hour, self.return_time.min])
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

                    trip_steps = ((ret_date - dep_date).total_seconds()
                                  / 3600 * 4)
                    total_distance = np.random.normal(
                        self.distance_travelled.mean_we,
                        self.distance_travelled.dev_we)
                    total_distance = max(
                        [total_distance, self.departure_time.min])
                    total_distance = min(
                        [total_distance, self.distance_travelled.max])
                    if total_distance < 0:
                        raise ValueError("Distance is negative")

                # Assuming no operation on Sundays

            # if trip is ongoing
            if (step >= dep_date) and (step < ret_date):

                # dividing the total distance into equal parts
                ev_schedule.loc[ev_schedule["date"] == step, "Distance_km"] = (
                        total_distance / trip_steps)

                # sampling consumption in kWh / km based on Emobpy German case statistics
                # Clipping to min
                cons_rating = max(
                    [np.random.normal(self.consumption.consumption_mean,
                                      self.consumption.consumption_std),
                     self.consumption.consumption_min])
                # Clipping to max
                cons_rating = min(
                    [cons_rating, self.consumption.consumption_max])
                # Clipping such that the maximum amount of energy per trip is not exceeded
                cons_rating = min(
                    [cons_rating,
                     self.consumption.total_cons_clip / total_distance])

                ev_schedule.loc[
                    ev_schedule["date"] == step, "Consumption_kWh"] = (
                        (total_distance / trip_steps) * cons_rating)

                # set relevant entries
                ev_schedule.loc[
                    ev_schedule["date"] == step, "Location"] = "driving"
                ev_schedule.loc[
                    ev_schedule["date"] == step, "ChargingStation"] = "none"
                ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(
                    ev_id)
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
                    ev_id)
                ev_schedule.loc[
                    ev_schedule["date"] == step, "PowerRating_kW"] = (
                    self.charger.charging_power)

        return ev_schedule

    def _generate_vec(self, ev_id: int) -> pd.DataFrame:
        # # make DataFrame and a date range, from start to end
        # ev_schedule = pd.DataFrame()
        # ev_schedule["date"] = pd.date_range(start=self.starting_date,
        #                                     end=self.ending_date,
        #                                     freq=self.freq)
        #
        # ev_schedule.set_index("date", inplace=True)
        #
        # if ev_schedule["date"][0].weekday() == 6:
        #     logger.info(
        #         "'starting_date' is a Sunday, "
        #         "generating schedule from next day onwards")
        #
        #     # TODO pandas code
        #     while ev_schedule["date"][0].weekday() == 6:
        #         ev_schedule.drop(index=0, inplace=True)
        #
        #     assert ev_schedule["date"][
        #                0].weekday != 6, "Error, first day is still a Sunday."
        #
        #     new_start = ev_schedule["date"][0]
        #     logger.info(f"Now starting on date: {new_start}")
        # if ev_schedule["date"][0].hour != 0 or ev_schedule["date"][0].minute != 0:
        #     logger.info(
        #         "'starting_date' is not at midnight, "
        #         "generating schedule from next day onwards"
        #     )
        #
        #     # TODO advance to next day
        #     pass
        #
        # for step in ev_schedule["date"]:
        #     # get variables at start of day
        #     # write values for trip (ongoing vs at home)
        #         # vectorisation (TODO)
        #         # df[~mask] = (0, 0, home, 11 kW)
        #         # df[mask] = (sample(), sample(), none, none)
        #         # create mask for trip and != trip
        #         # set != trip to default values
        #         # check if trip mask can be vectorised (random sampling)
        #
        # dep_date = None
        # ret_date = None
        # trip_mask = (ev_schedule["date"] >= dep_date) and (ev_schedule["date"] < ret_date)
        #
        # # create a mask which is True for weekdays
        # mask = ev_schedule.resample("1D").first()["date"].weekday() in [1, 2, 3, 4, 5]
        # num_days_wd = np.sum(mask)
        # num_days_we = len(mask) - num_days_wd
        #
        # dep_time_wd = np.random.normal(self.departure_time.mean_wd,
        #                                 self.departure_time.dev_wd,
        #                                 size=(num_days_wd,))
        # dep_time_we = np.random.normal(self.departure_time.mean_we,
        #                                self.departure_time.dev_we,
        #                                size=(num_days_we,))
        #
        #
        #
        # def interleave(a, b, mask):
        #     c = np.empty((a.size + b.size), dtype=a.dtype)
        #     c[mask] = a
        #     c[~mask] = b
        #     return c
        #
        #
        # dep_time = interleave(dep_time_wd, dep_time_we, mask)
        #
        # # min dep = 5
        # # min ret = 12
        # # max ret = 18
        # # delta = 6
        # # np[5, 5, 5, 10, 11, 9] + random(0, (max - min))
        #
        # start_date = ev_schedule["date"][0]
        # start_date.hour = self.departure_time.min
        # start_date.minutes = 0
        # dep_time_wd = np.full(num_days_wd, start_date + np.arange(num_days_wd))
        #
        # """
        # import numpy as np
        #
        # # Step 1: Create the array of datetime entries
        # N = 10  # Number of entries
        # start_date = np.datetime64('2023-01-01T00:00')
        # datetime_array = np.full(N, start_date)
        #
        # # Step 2: Define the resolution and generate random offsets
        # resolution = np.timedelta64(15, 'm')  # 15 minutes resolution
        # max_offsets = 24 * 60 // 15  # Number of 15 minute intervals in a day
        # random_offsets = np.random.randint(0, max_offsets, size=N) * resolution
        #
        # # Step 3: Add random offsets to the datetime entries
        # random_datetime_array = datetime_array + random_offsets
        #
        # # Print the result
        # print(random_datetime_array)
        # """
        #
        # dep_hour = np.modf(dep_time, dtype=int)[1]
        # dep_hour = np.clip(
        #     dep_hour, self.departure_time.min, self.departure_time.max)
        # # TODO make dynamic
        # minutes = np.asarray([0, 15, 30, 45])
        # # split number and decimals, use decimals and choose the closest minute
        # closest_index = np.abs(
        #     minutes - int(math.modf(dep_time)[0] * 60)).argmin()
        # dep_min = minutes[closest_index]
        #
        # # np.vstack((stats_dep_wd, stats_dep_we))
        #
        # # a = ev_schedule["date"].apply(rule_wd if wd == ... else rule_we)
        #
        # # a = wd[wd_mask] + we[we_mask]
        #
        # """
        # day weekday stats
        # 1   2 stats_wd
        # 2   3 stats_wd
        # 3   4 stats_wd
        # 4   5 stats_we
        # 5   6 stats_we
        # """
        #
        #
        # # Loop through each date entry and create the other entries
        # for step in ev_schedule["date"]:
        #
        #     # if new day, specify new random values
        #     if (step.hour == 0) and (step.minute == 0):
        #
        #         # weekdays
        #         if step.weekday() < 5:
        #
        #             # time mean and std dev in config
        #             dep_time = np.random.normal(self.departure_time.mean_wd,
        #                                         self.departure_time.dev_wd)
        #             # split number and decimals, use number and turn to int
        #             dep_hour = int(math.modf(dep_time)[1])
        #             dep_hour = min([dep_hour, self.departure_time.max])
        #             dep_hour = max([dep_hour, self.departure_time.min])
        #             # TODO make dynamic
        #             minutes = np.asarray([0, 15, 30, 45])
        #             # split number and decimals, use decimals and choose the closest minute
        #             closest_index = np.abs(
        #                 minutes - int(math.modf(dep_time)[0] * 60)).argmin()
        #             dep_min = minutes[closest_index]
        #
        #             ret_time = np.random.normal(self.return_time.mean_wd,
        #                                         self.return_time.dev_wd)
        #             ret_hour = int(math.modf(ret_time)[1])
        #             # clip return to a maximum
        #             ret_hour = min(ret_hour, self.return_time.max)
        #             ret_hour = max(ret_hour, self.return_time.min)
        #             minutes = np.asarray([0, 15, 30, 45])
        #             closest_index = np.abs(
        #                 minutes - int(math.modf(ret_time)[0] * 60)).argmin()
        #             ret_min = minutes[closest_index]
        #
        #             # make dates for easier comparison
        #             dep_date = datetime(step.year, step.month, step.day,
        #                                 hour=dep_hour, minute=dep_min)
        #             ret_date = datetime(step.year, step.month, step.day,
        #                                 hour=ret_hour, minute=ret_min)
        #
        #             # amount of time steps per trip
        #             trip_steps = ((ret_date - dep_date).total_seconds()
        #                           / 3600 * 4)
        #
        #             # total distance travelled that day
        #             total_distance = np.random.normal(
        #                 self.distance_travelled.mean_wd,
        #                 self.distance_travelled.dev_wd)
        #             total_distance = max(
        #                 [total_distance, self.distance_travelled.min])
        #             total_distance = min(
        #                 [total_distance, self.distance_travelled.max])
        #             if total_distance < 0:
        #                 raise ValueError("Distance is negative")
        #
        #         # weekend
        #         # elif step.weekday() == 5:
        #         #     dep_time = np.random.normal(self.departure_time.dep_mean_we,
        #         #                                 self.departure_time.dep_dev_we)
        #         #     dep_hour = int(math.modf(dep_time)[1])
        #         #     dep_hour = min([dep_hour, self.departure_time.max])
        #         #     dep_hour = max([dep_hour, self.departure_time.min])
        #         #     minutes = np.asarray([0, 15, 30, 45])
        #         #     closest_index = np.abs(
        #         #         minutes - int(math.modf(dep_time)[0] * 60)).argmin()
        #         #     dep_min = minutes[closest_index]
        #         #
        #         #     ret_time = np.random.normal(self.sc.ret_mean_we,
        #         #                                 self.sc.ret_dev_we)
        #         #     ret_hour = int(math.modf(ret_time)[1])
        #         #     # clip return to a maximum
        #         #     ret_hour = min([ret_hour, self.departure_time.max])
        #         #     ret_hour = max([ret_hour, self.departure_time.min])
        #         #     minutes = np.asarray([0, 15, 30, 45])
        #         #     closest_index = np.abs(
        #         #         minutes - int(math.modf(ret_time)[0] * 60)).argmin()
        #         #     ret_min = minutes[closest_index]
        #         #
        #         #     dep_date = dt.datetime(step.year, step.month, step.day,
        #         #                            hour=dep_hour, minute=dep_min)
        #         #     ret_date = dt.datetime(step.year, step.month, step.day,
        #         #                            hour=ret_hour, minute=ret_min)
        #         #
        #         #     if dep_date > ret_date:
        #         #         raise RuntimeError(
        #         #             "Schedule statistics produce unrealistic schedule. dep > ret.")
        #         #
        #         #     trip_steps = (
        #         #                          ret_date - dep_date).total_seconds() / 3600 * 4
        #         #     total_distance = np.random.normal(self.sc.avg_distance_we,
        #         #                                       self.sc.dev_distance_we)
        #         #     total_distance = max(
        #         #         [total_distance, self.sc.min])
        #         #     total_distance = min(
        #         #         [total_distance, self.sc.max])
        #         #     if total_distance < 0:
        #         #         raise ValueError("Distance is negative")
        #
        #         # Assuming no operation on Sundays
        #
        #     # if trip is ongoing
        #     if (step >= dep_date) and (step < ret_date):
        #
        #         # dividing the total distance into equal parts
        #         ev_schedule.loc[ev_schedule[
        #                             "date"] == step, "Distance_km"] = total_distance / trip_steps
        #
        #         # sampling consumption in kWh / km based on Emobpy German case statistics
        #         # Clipping to min
        #         cons_rating = max([np.random.normal(self.sc.consumption_mean,
        #                                             self.sc.consumption_std),
        #                            self.sc.consumption_min])
        #         # Clipping to max
        #         cons_rating = min([cons_rating, self.sc.consumption_max])
        #         # Clipping such that the maximum amount of energy per trip is not exceeded
        #         cons_rating = min(
        #             [cons_rating, self.sc.total_cons_clip / total_distance])
        #
        #         ev_schedule.loc[
        #             ev_schedule["date"] == step, "Consumption_kWh"] = (
        #                                                                       total_distance / trip_steps) * cons_rating
        #
        #         # set relevant entries
        #         ev_schedule.loc[
        #             ev_schedule["date"] == step, "Location"] = "driving"
        #         ev_schedule.loc[
        #             ev_schedule["date"] == step, "ChargingStation"] = "none"
        #         ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(
        #             ev_id)
        #         ev_schedule.loc[
        #             ev_schedule["date"] == step, "PowerRating_kW"] = 0.0
        #
        #     else:
        #         ev_schedule.loc[
        #             ev_schedule["date"] == step, "Distance_km"] = 0.0
        #         ev_schedule.loc[
        #             ev_schedule["date"] == step, "Consumption_kWh"] = 0.0
        #         ev_schedule.loc[
        #             ev_schedule["date"] == step, "Location"] = "home"
        #         ev_schedule.loc[
        #             ev_schedule["date"] == step, "ChargingStation"] = "home"
        #         ev_schedule.loc[ev_schedule["date"] == step, "ID"] = str(
        #             ev_id)
        #         ev_schedule.loc[ev_schedule[
        #                             "date"] == step, "PowerRating_kW"] = self.sc.charging_power
        #
        # return ev_schedule
        pass
