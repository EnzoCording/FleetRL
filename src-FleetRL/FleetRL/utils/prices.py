import pandas as pd


def load_prices(self):

    # load csv
    spot = pd.read_csv(self.path_name + "spot_2020.csv", delimiter=";", decimal=",")

    # drop price information of other countries
    spot = spot.drop(columns=spot.columns[4:20])

    # put the date in the same format as the schedule
    spot = spot.rename(columns={"Date": "date"})
    spot["date"] = spot["date"] + " " + spot["Start"] + ":00"
    spot["date"] = pd.to_datetime(spot["date"])

    # rename column for accessibility
    spot = spot.rename(columns={"Deutschland/Luxemburg [€/MWh] Original resolutions": "DELU"})

    # TODO test if this also works for down-sampling. Right now this up-samples from hourly to quarter-hourly
    self.spot_price = pd.merge_asof(self.date_range,
                                    spot.sort_values("date"),
                                    on="date",
                                    direction="backward"
                                    )

    # return the spot price at the right granularity
    return self.spot_price


def load_building_load(self):
    b_load = pd.read_csv(self.path_name + "building_load.csv", delimiter=",", decimal=",")

    # TODO test if this also works for down-sampling. Right now this up-samples from hourly to quarter-hourly
    self.building_load = pd.merge_asof(self.date_range,
                                       b_load.sort_values("date"),
                                       on="date",
                                       direction="backward"
                                       )

    # return building load at right granularity
    return self.building_load


def load_pv(self):
    pv = pd.read_csv(self.path_name + "pv_gen.csv", delimiter=",", decimal=",")

    self.pv_gen = pd.merge_asof(self.date_range,
                                pv.sort_values("date"),
                                on="date",
                                direction="backward")

    # return pv generation
    return self.pv_gen
