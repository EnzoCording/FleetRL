import rainflow
import numpy as np

from FleetRL.utils.new_battery_degradation.new_batt_deg import NewBatteryDegradation
from FleetRL.fleet_env.config.time_config import TimeConfig

import rainflow as rf

class NewRainflowSeiDegradation(NewBatteryDegradation):

    def __init__(self, init_soh: list):

        # Source: Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment
        # https://ieeexplore.ieee.org/document/7488267

        # initial state of health of the battery
        self.soh = init_soh

        # battery life, according to the paper notation
        self.l = np.ones(len(self.soh)) - self.soh

        # non-linear degradation model
        self.alpha_sei = 5.75E-2
        self.beta_sei = 121

        # DoD stress model
        self.kd1 = 1.4E5
        self.kd2 = -5.01E-1
        self.kd3 = -1.23E5

        # SoC stress model
        self.k_sigma = 1.04
        self.sigma_ref = 0.5

        # Temperature stress model
        self.k_temp = 6.93E-2
        self.temp_ref = 25  # °C

        # Calendar aging model
        self.k_dt = 4.14E-10  # 1/s -> per second

        # rainflow list counter to check when to calculate next degradation
        self.rainflow_length = np.ones(len(self.soh))

    def stress_dod(self, dod): return (self.kd1 * (dod ** self.kd2) + self.kd3) ** -1

    def stress_soc(self, soc): return np.e ** (self.k_sigma * (soc - self.sigma_ref))

    def stress_temp(self, temp): return np.e ** (self.k_temp * (temp - self.temp_ref) * (self.temp_ref / temp))

    def stress_time(self, t): return self.k_dt * t

    def deg_rate_cycle(self, dod, avg_soc, temp): return (self.stress_dod(dod)
                                                          * self.stress_soc(avg_soc)
                                                          * self.stress_temp(temp))

    def deg_rate_calendar(self, t, avg_soc, temp): return (self.stress_time(t)
                                                           * self.stress_soc(avg_soc)
                                                           * self.stress_temp(temp))

    def deg_rate_total(self, dod, avg_soc, temp, t): return (self.deg_rate_cycle(dod, avg_soc, temp)
                                                             + self.deg_rate_calendar(t, avg_soc, temp))

    def deg_rate_sei(self, dod, avg_soc, temp, t): return (self.beta_sei
                                                           * self.deg_rate_total(dod, avg_soc, temp, t))

    def l_with_sei(self, l, dod, avg_soc, temp, t): return (1 - self.alpha_sei * np.e ** (-1 * self.deg_rate_sei(dod, avg_soc, temp, t))
                                     - (1 - self.alpha_sei) * np.e ** (-1 * self.deg_rate_total(dod, avg_soc, temp, t)))

    def l_without_sei(self, l, dod, avg_soc, temp, t): return (1 - (1 - l) * np.e ** (-1 * self.deg_rate_total(dod, avg_soc, temp, t)))

    def calculate_degradation(self, soc_log: list, charging_power: float, time_conf: TimeConfig, temp: float) -> np.array:

        # compute sorted soc list based on the log records of the episode so far
        # go from: t1:[soc_car1, soc_car2, ...], t2:[soc_car1, soc_car2,...]
        # to this: car 1: [soc_t1, soc_t2, ...], car 2: [soc_t1, soc_t2, ...]
        sorted_soc_list = []

        # range(len(soc_log[0])) gives the number of cars
        for j in range(len(soc_log[0])):

            # range(len(soc_log)) gives the number of time steps that the cars go through
            sorted_soc_list.append([soc_log[i][j] for i in range(len(soc_log))])

        # this is 0 in the beginning and then gets updated with the new degradation due to the current time step
        degradation = np.zeros(len(sorted_soc_list))

        # calculate rainflow list and store it somewhere
        # check its length and see if it increased by one
        # if it increased, calculate with the previous entry, otherwise pass
        # I need the 0.5 / 1.0, the start and end, the average, the DoD

        # len(sorted_soc_list) gives the number of cars
        for i in range(len(sorted_soc_list)):

            # empty list, then fill it with the results of rainflow and check if it got longer
            rainflow_result = []

            # execute rainflow counting algorithm
            for rng, mean, count, i_start, i_end in rainflow.extract_cycles(sorted_soc_list[i]):
                rainflow_result.append([rng, mean, count, i_start, i_end])

            # check if a new entry appeared in the results of the rainflow counting
            if len(rainflow_result) > self.rainflow_length[i]:

                # calculate degradation of the 2nd last rainflow entry (the most recent might still change)
                last_complete_entry = rainflow_result[-2]

                # dod is equal to the range
                dod = last_complete_entry[0]

                # average soc is equal to the mean
                avg_soc = last_complete_entry[1]

                # severity is equal to count: either 0.5 or 1.0
                degradation_severity = last_complete_entry[2]

                # half or full cycle
                effective_dod = dod * degradation_severity

                # time of the cycle
                t = (last_complete_entry[4] - last_complete_entry[3]) * time_conf.dt * 3600

                # check if L is smaller than 0.1 -> sei film formation ongoing
                if self.l[i] < 0.1:
                    new_l = self.l_with_sei(self.l[i], effective_dod, avg_soc, temp, t)

                    # check if l is negative, then something is wrong
                    if new_l < 0:
                        raise TypeError("Life degradation is negative")

                # if L bigger than 0.1, sei film formation is done
                else:
                    new_l = self.l_without_sei(self.l[i], effective_dod, avg_soc, temp, t)
                    print("Using the other way")
                print(f"new_l: {new_l}")

                degradation[i] = new_l - self.l[i]

                self.l[i] = new_l
                self.rainflow_length[i] = len(rainflow_result)

            else:
                # degradation = 0
                lifetime = self.l

        self.l = lifetime
        self.soh -= degradation
        print(f"soh sei: {self.soh}")

        return np.array(lifetime)
