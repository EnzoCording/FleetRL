import datetime

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
                 vehicle_id: int = 0):

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

        # make DataFrame and a date range, from start to end
        ev_schedule = pd.DataFrame()
        ev_schedule["date"] = pd.date_range(start=self.starting_date, end=self.ending_date, freq = self.freq)

        # Loop through each date entry and create the other entries
        for step in ev_schedule["date"]:

            # if new day, specify new random values
            if step.hour == 0:

                # weekdays
                if step.weekday() < 5:

                    # time mean and std dev in config
                    dep_hour = int(round(np.random.normal(self.sc.dep_mean_wd, self.sc.dep_dev_wd), 0))
                    dep_min = np.random.choice([0, 15, 30, 45])

                    ret_hour = int(round(np.random.normal(self.sc.ret_mean_wd, self.sc.ret_dev_wd), 0))
                    ret_min = np.random.choice([0, 15, 30, 45])

                    # make dates for easier comparison
                    dep_date = dt.datetime(step.year, step.month, step.day, hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day, hour=ret_hour, minute=ret_min)

                    # amount of time steps per trip
                    trip_steps = (ret_date - dep_date).total_seconds() / 3600 * 4

                    # total distance travelled that day
                    total_distance = np.random.normal(self.sc.avg_distance_wd, self.sc.dev_distance_wd)

                # weekend
                else:
                    dep_hour = int(round(np.random.normal(self.sc.dep_mean_we, self.sc.dep_dev_we), 0))
                    dep_min = np.random.choice([0, 15, 30, 45])

                    ret_hour = int(round(np.random.normal(self.sc.ret_mean_we, self.sc.ret_dev_we), 0))
                    ret_min = np.random.choice([0, 15, 30, 45])

                    dep_date = dt.datetime(step.year, step.month, step.day, hour=dep_hour, minute=dep_min)
                    ret_date = dt.datetime(step.year, step.month, step.day, hour=ret_hour, minute=ret_min)

                    trip_steps = (ret_date - dep_date).total_seconds() / 3600 * 4
                    total_distance = np.random.normal(self.sc.avg_distance_we, self.sc.dev_distance_we)

            # if trip is ongoing
            if (step >= dep_date) and (step < ret_date):

                # dividing the total distance into equal parts
                ev_schedule.loc[ev_schedule["date"] == step, "Distance_km"] = total_distance / trip_steps

                # sampling consumption in kWh / km based on Emobpy German case statistics
                cons_rating = max([np.random.normal(self.sc.consumption_mean, self.sc.consumption_std),
                                   self.sc.consumption_min])
                ev_schedule.loc[ev_schedule["date"] == step, "Consumption_kWh"] = total_distance / trip_steps * cons_rating

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
                ev_schedule.loc[ev_schedule["date"] == step, "PowerRating_kW"] = 11.0

        if self.save:
            ev_schedule.to_csv(self.path_name)

        return ev_schedule

    def generate_multiple_ev_schedule(self, num_evs):
        schedule = pd.DataFrame()
        self.save = False
        self.file_comment += f"_{num_evs}_evs"
        self.file_name = f"schedule_{self.time_now}_{self.file_comment}.csv"
        self.path_name = self.schedule_dir + self.file_name
        for i in range(num_evs):
            self.vehicle_id = i
            new_schedule = self.generate_schedule()
            schedule = pd.concat([schedule, new_schedule], axis = 0, ignore_index=True)
        schedule.to_csv(self.path_name)
