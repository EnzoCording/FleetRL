import pandas as pd


def load_prices(self):
    spot = pd.read_csv(self.path_name + "spot_2020.csv", delimiter=";", decimal=",")
    spot["Date"] = spot["Date"] + " " + spot["Start"] + ":00"
    spot["date"] = pd.to_datetime(spot["Date"])
    spot = spot.drop(columns=["Date"])
    spot = spot.drop(columns=spot.columns[3:19])

    spot = spot.rename(columns={"Deutschland/Luxemburg [€/MWh] Original resolutions": "DELU"})

    self.spot_price = pd.merge_asof(self.date_range.sort_values("date"),
                                    spot.sort_values("date"),
                                    on="date",
                                    direction="backward"
                                    )



    return self.spot_price
