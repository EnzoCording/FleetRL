import datetime

import numpy as np
import pandas as pd

from FleetRL.fleet_env.config.ev_config import EvConfig
from FleetRL.fleet_env.config.time_config import TimeConfig


# this class contains all the necessary information from the vehicle and its schedule
# the schedule is imported from emobpy and manipulated so that the output data is in the right format

class DataLoader:

    def __init__(self, path_name, schedule_name, spot_name, building_name, pv_name,
                 time_conf: TimeConfig, ev_conf: EvConfig,
                 target_soc, building_flag, pv_flag):

        # save the time_conf within DataLoader as well because it is used in some functions
        self.time_conf = time_conf

        # schedule import from excel
        # db = pd.read_excel(os.path.dirname(__file__) + '/test_simple.xlsx')
        self.schedule = pd.read_csv(path_name + schedule_name, parse_dates=["date"])

        # setting the index of the df to the date for resampling
        self.schedule.set_index("date", inplace=True, drop=False)

        # resampling the df. consumption and distance are summed, power rating mean like in emobpy
        # group by ID is needed so the different cars don't get overwritten (they have the same dates)
        # NB: up-sampling is not going to work
        self.schedule = self.schedule.groupby("ID").resample(time_conf.freq).agg(
            {'Location': 'first', 'ID': 'first', 'Consumption_kWh': 'sum',
             'ChargingStation': 'first', 'PowerRating_kW': 'mean',
             'Distance_km': 'sum', 'date': 'first'})

        # resetting the index to a numerical value
        self.schedule.index = range(len(self.schedule))

        self.compute_from_schedule(ev_conf, time_conf, target_soc)

        # create a date range with the chosen frequency
        # Given the desired frequency, create a (down-sampled) column of timestamps
        self.date_range = pd.DataFrame()
        self.date_range["date"] = pd.date_range(start=self.schedule["date"].min(),
                                                end=self.schedule["date"].max(),
                                                freq=time_conf.freq
                                                )

        self.spot_price = DataLoader.load_prices(path_name, spot_name, self.date_range)

        if building_flag:
            self.building_load = DataLoader.load_building_load(path_name, building_name, self.date_range)

        if pv_flag:
            self.pv = DataLoader.load_pv(path_name, pv_name, self.date_range)

        if not building_flag and not pv_flag:
            self.db = pd.concat([self.schedule, self.spot_price["DELU"]], axis=1)

        elif building_flag and not pv_flag:
            self.db = pd.concat([self.schedule, self.spot_price["DELU"], self.building_load["load"]], axis=1)

        elif not building_flag and pv_flag:
            self.db = pd.concat([self.schedule, self.spot_price["DELU"], self.pv["pv"]], axis=1)

        elif building_flag and pv_flag:
            self.db = pd.concat([self.schedule, self.spot_price["DELU"], self.building_load["load"],
                                 self.pv["pv"]], axis=1)

        else:
            self.db = None
            raise RuntimeError("Problem with components. Check building and PV flags.")

    def compute_from_schedule(self, ev_conf, time_conf, target_soc):
        """
        # >>> test_EV = EV()
        # >>> test_EV.get_total_consumption().sum().round(1)
        # 147.8
    
        :return:
        """
        # new column with flag if EV is there or not
        self.schedule["There"] = (np.array(self.schedule["PowerRating_kW"] != 0, dtype=int))

        # make a new column in db that checks whether the charging station value changed
        self.schedule["Change"] = np.array(self.schedule["ChargingStation"] != self.schedule["ChargingStation"].shift(1), dtype=int)

        # create a group, so they can be easily grouped by change (home->none or none->home)
        self.schedule["Group"] = self.schedule["Change"].cumsum()

        # create a column for the total consumption that will later be populated
        self.schedule["TotalConsumption"] = np.zeros(len(self.schedule))

        # calculate the total consumption of a single trip by summing over a group
        # only choose the groups where the car is driving (=="none")
        # sum over the consumption
        # resetting the index and dropping the old group index
        consumption = (self.schedule.loc[self.schedule["ChargingStation"] == "none"]
                       .groupby('Group')["Consumption_kWh"].sum().reset_index(drop=True))

        trip_length = (self.schedule.loc[self.schedule["ChargingStation"] == "none"]
                       .groupby('Group')["date"].count().reset_index(drop=True))

        # get the last dates of each trip
        # resetting the index so the group index goes away
        last_dates = (self.schedule.loc[self.schedule["ChargingStation"] == "none"]
                      .groupby('Group')["date"].last().reset_index(drop=True))

        # get the first dates of each trip
        # resetting the index so the group index goes away
        departure_dates = (self.schedule.loc[self.schedule["ChargingStation"] == "none"]
                           .groupby('Group')["date"].first().reset_index(drop=True))

        # the return dates are on the next timestep
        # drop duplicates because the date is iterated through for each ev anyway
        return_dates = last_dates.add(datetime.timedelta(minutes=time_conf.minutes))
        # get the vehicle ids of the trips
        # resetting the index so the group index goes away
        # dropping duplicates because the loop iterates through both dates and ids anyway
        ids = self.schedule.loc[self.schedule["ChargingStation"] == "none"].groupby('Group')["ID"].first().reset_index(drop=True)

        # creating a dataframe for calculating the consumption, the info of ID and date is needed
        res_return = pd.DataFrame({"ID": ids, "consumption": consumption, "date": return_dates, "len": trip_length})

        # creating a dataframe for calculating the time_left, ID and date needed for pd.merge_asof()
        res_departure = pd.DataFrame({"ID": ids, "dep": departure_dates, "date": departure_dates})

        # match return dates with db, backwards fill, sort by ID, match on date
        merged_cons = pd.merge_asof(self.schedule.sort_values("date"),
                                    res_return.sort_values("date"),
                                    on="date",
                                    by="ID",
                                    direction="backward"
                                    )

        # sort the df into the right order and reset the index
        merged_cons = merged_cons.sort_values(["ID", "date"]).reset_index(drop=True)

        # set consumption to 0 where the car is not there because information is unknown to the agent at that point
        merged_cons.loc[merged_cons["There"] == 0, "consumption"] = 0

        # set trip length to 0 where the car is not there because information unknown to agent at that point
        merged_cons.loc[merged_cons["There"] == 0, "len"] = 0

        # fill NaN values
        merged_cons.fillna(0, inplace=True)

        # match departure dates with dates in db, forward direction, sort by ID, match on date
        merged_time_left = pd.merge_asof(self.schedule.sort_values("date"),
                                         res_departure.sort_values("date"),
                                         on="date",
                                         by="ID",
                                         direction="forward"
                                         )

        # reset the index
        merged_time_left = merged_time_left.sort_values(["ID", "date"]).reset_index(drop=True)
        # calculate time left with the correct index
        merged_time_left["time_left"] = (merged_time_left["dep"] - merged_time_left["date"]).dt.total_seconds() / 3600
        # time left is 0 when the car is not there
        merged_time_left.loc[merged_time_left["There"] == 0, "time_left"] = 0
        # fill NaN values
        merged_time_left.loc[:, "time_left"].fillna(0, inplace=True)

        # add computed information to db
        self.schedule["last_trip_total_consumption"] = merged_cons.loc[:, "consumption"]
        self.schedule["last_trip_total_length_hours"] = merged_cons.loc[:, "len"].div(self.time_conf.time_steps_per_hour)
        self.schedule["time_left"] = merged_time_left.loc[:, "time_left"]

        # create BOC column and populate with zeros
        # calculate SOC on return, assuming the previous trip charged to the target soc
        # TODO this could be changed in the future to make it more complex
        self.schedule["SOC_on_return"] = target_soc - self.schedule["last_trip_total_consumption"].div(
            ev_conf.battery_cap)
        # TODO could be set to -1
        self.schedule.loc[self.schedule["There"] == 0, "SOC_on_return"] = 0

    @staticmethod
    def load_prices_original(path_name, spot_name, date_range):
        # load csv
        spot = pd.read_csv(path_name + spot_name, delimiter=";", decimal=",")

        # drop price information of other countries
        spot = spot.drop(columns=spot.columns[4:20])

        # put the date in the same format as the schedule
        spot = spot.rename(columns={"Date": "date"})
        spot["date"] = spot["date"] + " " + spot["Start"] + ":00"
        spot["date"] = pd.to_datetime(spot["date"], format="mixed")

        # rename column for accessibility
        spot = spot.rename(columns={"Deutschland/Luxemburg [€/MWh] Original resolutions": "DELU"})

        # TODO test if this also works for down-sampling. Right now this up-samples from hourly to quarter-hourly
        spot_price = pd.merge_asof(date_range,
                                   spot.sort_values("date"),
                                   on="date",
                                   direction="backward"
                                   )

        # return the spot price at the right granularity
        return spot_price

    @staticmethod
    def load_prices(path_name, spot_name, date_range):
        # load csv
        spot = pd.read_csv(path_name + spot_name, delimiter=";", decimal=",", parse_dates=["date"])

        # drop price information of other countries
        spot = spot.drop(columns=spot.columns[4:20])

        # put the date in the same format as the schedule
        # spot["date"] = pd.to_datetime(spot["date"], format="mixed")

        # rename column for accessibility
        spot = spot.rename(columns={"Deutschland/Luxemburg [€/MWh] Original resolutions": "DELU"})

        # TODO test if this also works for down-sampling. Right now this up-samples from hourly to quarter-hourly
        spot_price = pd.merge_asof(date_range,
                                   spot.sort_values("date"),
                                   on="date",
                                   direction="backward"
                                   )

        # return the spot price at the right granularity
        return spot_price

    @staticmethod
    def load_building_load(path_name, file_name, date_range):
        b_load = pd.read_csv(path_name + file_name, delimiter=",", parse_dates=["date"])
        # b_load["date"] = pd.to_datetime(b_load["date"], format="mixed")

        # TODO test if this also works for down-sampling. Right now this up-samples from hourly to quarter-hourly
        building_load = pd.merge_asof(date_range,
                                      b_load.sort_values("date"),
                                      on="date",
                                      direction="backward"
                                      )

        # return building load at right granularity
        return building_load

    @staticmethod
    def load_pv(path_name, pv_name, date_range):
        pv = pd.read_csv(path_name + pv_name, delimiter=",", decimal=",", parse_dates=["date"])
        # pv["date"] = pd.to_datetime(pv["date"], format="mixed")

        pv["pv"] = pv["pv"].astype(float)

        pv = pd.merge_asof(date_range,
                           pv.sort_values("date"),
                           on="date",
                           direction="backward")

        # return pv generation
        return pv
