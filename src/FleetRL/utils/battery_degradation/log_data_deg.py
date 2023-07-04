import pandas as pd
import time
import copy

class LogDataDeg:
    def __init__(self, episode):
        self.log: list = []
        self.soc_log = []
        self.soh_log = []

    def log_soc(self, soc):
        self.soc_log.append(copy.deepcopy(soc))

    def log_soh(self, soh):
        self.soh_log.append(copy.deepcopy(soh))

    def add_log_entry(self):
        self.log.append({"soc": copy.deepcopy(self.soc_log),
                         "soh": copy.deepcopy(self.soh_log)})  # , "econ": self.econ_list})
