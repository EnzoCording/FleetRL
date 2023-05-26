import pandas as pd

class DataLogger:
    def __init__(self, episode):
        self.log: list = []
        self.soc_log = []
        self.econ_list = []

    def log_soc(self, episode):
        self.soc_log.append(episode.soc)

    def add_log_entry(self):
        self.log.append({"soc": self.soc_log})  # , "econ": self.econ_list})
        #print("printing log:")
        #print(self.log)