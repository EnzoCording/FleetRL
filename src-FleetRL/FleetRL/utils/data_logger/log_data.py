import pandas as pd

class DataLogger:
    def __init__(self, episode):
        self.log: list = []
        self.soc_list = []
        self.econ_list = []

    def log_soc(self, episode):
        self.soc_list.append(episode.soc)

    def add_log_entry(self):
        self.log.append({"soc": self.soc_list, "econ": self.econ_list})
        print(self.log)