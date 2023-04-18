import pandas as pd
import numpy as np
import datetime as dt
import os
import random


# this class contains all the necessary information from the vehicle and its schedule
# the schedule is imported from emobpy and manipulated so that the output data is in the right format

def load_schedule(self):
    # schedule import from excel
    # self.db = pd.read_excel(os.path.dirname(__file__) + '/test_simple.xlsx')
    self.db = pd.read_csv(self.path_name + self.db_name, parse_dates=["date"])

    # setting the index of the df to the date for resampling
    self.db.set_index("date", inplace=True, drop=False)

    # resampling the df. consumption and distance are summed, power rating mean like in emobpy
    # group by ID is needed so the different cars don't get overwritten (they have the same dates)
    # TODO up-sampling is not going to work
    self.db = self.db.groupby("ID").resample(self.freq).agg(
        {'Location': 'first', 'ID': 'first', 'Consumption_kWh': 'sum',
         'ChargingStation': 'first', 'PowerRating_kW': 'mean',
         'Distance_km': 'sum', 'date': 'first'})

    # resetting the index to a numerical value
    self.db.index = range(len(self.db))

    # # creating a daterange for enabling loops through the dates
    # self.dr = pd.date_range(start=self.db["date"].iloc[0], end=self.db["date"].iloc[-1], freq=self.freq)
    return self.db


def compute_from_schedule(self):
    """
    # >>> test_EV = EV()
    # >>> test_EV.get_total_consumption().sum().round(1)
    # 147.8


    :return:
    """
    # new column with flag if EV is there or not
    self.db["There"] = (np.array(self.db["PowerRating_kW"] != 0, dtype=int))

    # make a new column in db that checks whether the charging station value changed
    self.db["Change"] = np.array(self.db["ChargingStation"] != self.db["ChargingStation"].shift(1), dtype=int)

    # create a group, so they can be easily grouped by change (home->none or none->home)
    self.db["Group"] = self.db["Change"].cumsum()

    # create a column for the total consumption that will later be populated
    self.db["TotalConsumption"] = np.zeros(len(self.db))

    # calculate the total consumption of a single trip by summing over a group
    # only choose the groups where the car is driving (=="none")
    # sum over consumption
    # resetting the index and dropping the old group index
    consumption = (self.db.loc[self.db["ChargingStation"] == "none"]
                   .groupby('Group')["Consumption_kWh"].sum().reset_index(drop=True))

    # get the last dates of each trip
    # resetting the index so the group index goes away
    last_dates = (self.db.loc[self.db["ChargingStation"] == "none"]
                  .groupby('Group')["date"].last().reset_index(drop=True))

    # get the first dates of each trip
    # resetting the index so the group index goes away
    departure_dates = (self.db.loc[self.db["ChargingStation"] == "none"]
                       .groupby('Group')["date"].first().reset_index(drop=True))

    # the return dates are on the next timestep
    # drop duplicates because the date is iterated through for each ev anyway
    return_dates = last_dates.add(dt.timedelta(minutes=self.minutes))

    # get the vehicle ids of the trips
    # resetting the index so the group index goes away
    # dropping duplicates because the loop iterates through both dates and ids anyway
    ids = self.db.loc[self.db["ChargingStation"] == "none"].groupby('Group')["ID"].first().reset_index(drop=True)

    # creating a dataframe for calculating the consumption, the info of ID and date is needed
    res_return = pd.DataFrame({"ID": ids, "consumption": consumption, "date": return_dates})

    # creating a dataframe for calculating the time_left, ID and date needed for pd.merge_asof()
    res_departure = pd.DataFrame({"ID": ids, "dep": departure_dates, "date": departure_dates})

    # match return dates with self.db, backwards fill, sort by ID, match on date
    merged_cons = pd.merge_asof(self.db.sort_values("date"),
                                res_return.sort_values("date"),
                                on="date",
                                by="ID",
                                direction="backward"
                                )

    # sort the df into the right order and reset the index
    merged_cons = merged_cons.sort_values(["ID", "date"]).reset_index(drop=True)

    # set consumption to 0 where the car is not there because information is unknown to the agent then
    merged_cons.loc[merged_cons["There"] == 0, "consumption"] = 0

    # fill NaN values
    merged_cons.fillna(0, inplace=True)

    # match departure dates with dates in self.db, forward direction, sort by ID, match on date
    merged_time_left = pd.merge_asof(self.db.sort_values("date"),
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

    # add computed information to self.db
    self.db["last_trip_total_consumption"] = merged_cons.loc[:, "consumption"]
    self.db["time_left"] = merged_time_left.loc[:, "time_left"]

    # create BOC column and populate with zeros
    # calculate SOC on return, assuming the previous trip charged to the target soc
    # TODO this could be changed in the future to make it more complex
    self.db["SOC_on_return"] = self.target_soc - self.db["last_trip_total_consumption"].div(self.battery_cap)
    self.db.loc[self.db["There"] == 0, "SOC_on_return"] = 0

    return self.db
