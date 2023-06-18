import pandas as pd
import time

class DataLogger:
    def __init__(self, episode):
        self.log: list = []
        self.soc_log = []
        self.soh_log = []
        self.econ_log = []

    def log_soc(self, episode):
        self.soc_log.append(episode.soc_deg.copy())

    def log_soh(self, episode):
        self.soh_log.append(episode.soh.copy())

    def add_log_entry(self):
        self.log.append({"soc": self.soc_log, "soh": self.soh_log})  # , "econ": self.econ_list})
        #print("printing log:")
        #print(self.log)

    def permanent_log(self):
        time_now = int(time.time())
        log = pd.DataFrame(self.log)
        log.to_csv(f"log_{time_now}.csv")