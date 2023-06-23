import pandas as pd
import time

class LogDataDeg:
    def __init__(self, episode):
        self.log: list = []
        self.soc_log = []
        self.soh_log = []

    def log_soc(self, soc):
        self.soc_log.append(soc)

    def log_soh(self, soh):
        self.soh_log.append(soh)

    def add_log_entry(self):
        self.log.append({"soc": self.soc_log, "soh": self.soh_log})  # , "econ": self.econ_list})
