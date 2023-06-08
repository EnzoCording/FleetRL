import rainflow
import numpy as np

from FleetRL.utils.new_battery_degradation.new_batt_deg import NewBatteryDegradation
from FleetRL.fleet_env.config.time_config import TimeConfig


class NewRainflowSeiDegradation(NewBatteryDegradation):

    def __init__(self, init_soh: float, num_cars: int):

        # Source: Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment
        # https://ieeexplore.ieee.org/document/7488267

        self.num_cars = num_cars
        self.adj_counter = 0

        # initial state of health of the battery
        self.init_soh = init_soh
        self.soh = np.ones(self.num_cars) * self.init_soh

        # battery life, according to the paper notation
        self.l = np.ones(self.num_cars) - self.soh

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
        self.rainflow_length = np.ones(self.num_cars)

        self.fd: np.array = np.zeros(self.num_cars)

    def stress_dod(self, dod): return (self.kd1 * (dod ** self.kd2) + self.kd3) ** -1

    def stress_soc(self, soc): return np.e ** (self.k_sigma * (soc - self.sigma_ref))

    def stress_temp(self, temp): return np.e ** (self.k_temp * (temp - self.temp_ref)
                                                 * ((self.temp_ref + 273.15) / (temp + 273.15)))

    def stress_time(self, t): return self.k_dt * t

    def deg_rate_cycle(self, dod, avg_soc, temp): return (self.stress_dod(dod)
                                                          * self.stress_soc(avg_soc)
                                                          * self.stress_temp(temp))

    def deg_rate_calendar(self, t, avg_soc, temp): return (self.stress_time(t)
                                                           * self.stress_soc(avg_soc)
                                                           * self.stress_temp(temp))

    def deg_rate_total(self, dod, avg_soc, temp, t): return (self.deg_rate_cycle(dod, avg_soc, temp)
                                                             + self.deg_rate_calendar(t, avg_soc, temp))

    def l_with_sei(self, car): return (1 - self.alpha_sei * np.e ** (-1 * self.beta_sei * self.fd[car])
                                  - (1 - self.alpha_sei) * np.e ** (-1 * self.fd[car]))

    def l_without_sei(self, l, car): return 1 - (1 - l) * np.e ** (-1 * self.fd[car])

    def calculate_degradation(self, soc_log: list, charging_power: float, time_conf: TimeConfig, temp: float) -> np.array:

        # compute sorted soc list based on the log records of the episode so far
        # go from: t1:[soc_car1, soc_car2, ...], t2:[soc_car1, soc_car2,...]
        # to this: car 1: [soc_t1, soc_t2, ...], car 2: [soc_t1, soc_t2, ...]

        sorted_soc_list = []

        for j in range(self.num_cars):

            # range(len(soc_log)) gives the number of time steps that the cars go through
            sorted_soc_list.append([soc_log[i][j] for i in range(len(soc_log))])

        # this is 0 in the beginning and then gets updated with the new degradation due to the current time step
        degradation = np.zeros(len(sorted_soc_list))

        # calculate rainflow list and store it somewhere
        # check its length and see if it increased by one
        # if it increased, calculate with the previous entry, otherwise pass
        # I need the 0.5 / 1.0, the start and end, the average, the DoD

        # len(sorted_soc_list) gives the number of cars
        for i in range(self.num_cars):

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

                # self.deg_rate_total becomes negative for DoD > 1
                # the two checks below count how many times dod is adjusted and in severe cases stops the code

                if (dod > 2) and (degradation_severity == 0.5):
                    self.adj_counter += 1
                    print("Minor adjustment made to DoD for degradation calculation.")

                if dod > 5:
                    print("Dod should be checked. Split cycle into multiple cycles.")
                    print("Remove this Error if problem should be ignored.")
                    raise TypeError("DoD too large.")

                # half or full cycle, max of 1
                effective_dod = min([1, dod * degradation_severity])

                # time of the cycle
                t = (last_complete_entry[4] - last_complete_entry[3]) * time_conf.dt * 3600

                # check if new battery, otherwise ignore sei film formation
                if self.init_soh == 1.0:
                    self.fd[i] += self.deg_rate_total(effective_dod, avg_soc, temp, t)
                    new_l = self.l_with_sei(i)

                    # check if l is negative, then something is wrong
                    if new_l < 0:
                        raise TypeError("Life degradation is negative")

                # if battery used, sei film formation is done and can be ignored
                else:
                    self.fd[i] += self.deg_rate_total(effective_dod, avg_soc, temp, t)
                    new_l = self.l_without_sei(self.l[i], i)

                # calculate degradation based on the change of l
                degradation[i] = new_l - self.l[i]

                # update lifetime variable
                self.l[i] = new_l

                # set new rainflow_length for this car
                self.rainflow_length[i] = len(rainflow_result)

            else:
                degradation[i] = 0

            if degradation[i] < 0:
                raise TypeError("Degradation negative, might have to do with DoD > 1.")

        self.soh -= degradation

        # check that the adding up of degradation is equivalent to the newest lifetime value calculated
        if abs(self.soh - (1 - self.l[i])) > 0.0001:
            raise RuntimeError("Degradation calculation is not correct")

        # print(f"sei soh: {self.soh}")

        return np.array(degradation)
