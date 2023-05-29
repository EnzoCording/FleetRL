import pandas as pd

class DataLogger:
    def __init__(self, episode):
        self.log: list = []
        self.soc_log = []
        self.soh_log = []
        self.soh_2 = []
        self.econ_log = []

    def log_soc(self, episode):
        self.soc_log.append(episode.soc.copy())

    def log_soh(self, episode):
        self.soh_log.append(episode.soh.copy())
        self.soh_2.append(episode.soh_2.copy())

    def add_log_entry(self):
        self.log.append({"soc": self.soc_log, "soh": self.soh_log, "soh2": self.soh_2})  # , "econ": self.econ_list})
        #print("printing log:")
        #print(self.log)