import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rainflow as rf

class Comparison:

    def __init__(self):
        # Source: Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment
        # https://ieeexplore.ieee.org/document/7488267

        num_cars = 1

        self.num_cars = num_cars
        self.adj_counter = 0

        # initial state of health of the battery
        init_soh = 1.0
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

        # Accumulated function value for fd for cycles
        self.fd_cyc: np.array = np.zeros(self.num_cars)

        # fd value for calendar aging, is overwritten every iteration
        self.fd_cal: np.array = np.zeros(self.num_cars)

        # Absolute capacity reduction of the last cycle
        self.degradation = np.zeros(self.num_cars)

        self.cycle_loss_11 = 0.000125  # Cycle loss per full cycle (100% DoD discharge and charge) at 11 kW
        self.cycle_loss_22 = 0.000125  # Cycle loss per full cycle (100% DoD discharge and charge) at 11 kW
        self.cycle_loss_43 = 0.000167  # Cycle loss per full cycle (100% DoD discharge and charge) at 43 kW

        self.calendar_aging_0 = 0.0065  # Calendar aging per year if battery at 0% SoC
        self.calendar_aging_40 = 0.0293  # Calendar aging per year if battery at 40% SoC
        self.calendar_aging_90 = 0.065  # Calendar aging per year if battery at 90% SoC

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

    def l_with_sei(self, fd): return (1 - self.alpha_sei * np.e ** (-self.beta_sei * fd)
                                      - (1 - self.alpha_sei) * np.e ** (-fd))

    @staticmethod
    def l_without_sei(self, l, fd): return 1 - (1 - l) * np.e ** (-fd)

    def compare_methods(self, data, years):
        plt.plot(self.rainflow_sei(data))
        plt.plot(self.emp_deg(data))
        plt.grid(True)
        plt.xlim([0, 10])
        plt.ylim([0.8, 1])
        plt.legend(["SEI formation + Cycle counting", "Empirical linear degradation"])
        plt.xticks(range(11))
        plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0])
        plt.savefig("sei_emp_10.pdf")
        plt.show()

    def rainflow_sei(self, soc_log):
        sorted_soc_list = soc_log

        # this is 0 in the beginning and then gets updated with the new degradation due to the current time step
        degradation = np.zeros(len(sorted_soc_list))

        # calculate rainflow list and store it somewhere
        # check its length and see if it increased by one
        # if it increased, calculate with the previous entry, otherwise pass
        # I need the 0.5 / 1.0, the start and end, the average, the DoD

        rainflow_df = pd.DataFrame(columns=['Range', 'Mean', 'Count', 'Start', 'End'])

        for rng, mean, count, i_start, i_end in rf.extract_cycles(np.tile(sorted_soc_list, 1)):
            new_row = pd.DataFrame(
                {'Range': [rng], 'Mean': [mean], 'Count': [count], 'Start': [i_start], 'End': [i_end]})
            rainflow_df = pd.concat([rainflow_df, new_row], ignore_index=True)

        L_sei = []

        # len(sorted_soc_list) gives the number of cars
        for i in range(1, 11):

            rainflow_data = rainflow_df.loc[rainflow_df.index.repeat((i) * 8760 * 4 / len(sorted_soc_list))]

            # dod is equal to the range
            dod = rainflow_data["Range"]

            # average soc is equal to the mean
            avg_soc = rainflow_data["Mean"]

            if len(avg_soc) == 0:
                mean_cal = soc_log.mean()
            else:
                mean_cal = avg_soc.mean()

            # severity is equal to count: either 0.5 or 1.0
            degradation_severity = rainflow_data["Count"]

            # deg_rate_total becomes negative for DoD > 1
            # the two checks below count how many times dod is adjusted and in severe cases stops the code

            # half or full cycle, max of 1
            effective_dod = np.clip(dod * degradation_severity, 0, 1)

            # time of the cycle
            t = (rainflow_data["End"] - rainflow_data["Start"]) * 0.25 * 3600
            t_cal = np.max(rainflow_data["End"]) * 0.25 * 3600

            f_cyc = self.deg_rate_cycle(effective_dod, avg_soc, self.temp_ref)
            f_cal = self.deg_rate_calendar(t_cal * i, avg_soc=mean_cal, temp=self.temp_ref)

            # check if new battery, otherwise ignore sei film formation
            if self.init_soh == 1.0:
                fd = f_cyc.sum() + f_cal
                new_l = self.l_with_sei(fd)

                # check if l is negative, then something is wrong
                if new_l < 0:
                    raise TypeError("Life degradation is negative")

            # if battery used, sei film formation is done and can be ignored
            else:
                "Not implemented yet"

            # calculate degradation based on the change of l
            degradation[0] = new_l - self.l[0]

            # update lifetime variable
            self.l[0] = new_l

            self.soh[0] -= degradation[0]

            L_sei = np.append(L_sei, self.l)

        # check that the adding up of degradation is equivalent to the newest lifetime value calculated
        if abs(self.soh[0] - (1 - self.l[0])) > 0.0001:
            raise RuntimeError("Degradation calculation is not correct")

        # print(f"sei soh: {soh}")

        print(L_sei)
        print(self.soh)

        L_sei = np.insert(L_sei, 0, 0)

        return np.subtract(1, L_sei)

    def emp_deg(self, data):
        data = np.array(data["soc"].values)
        sorted_soc_list = data
        charging_power = 11
        dt = 0.25
        soh_emp = []
        degradation = []

        for j in range(1, 11):

            for i in range(1, len(data)):

                old_soc = sorted_soc_list[i - 1]
                new_soc = sorted_soc_list[i]

                # compute average for calendar aging
                avg_soc = (old_soc + new_soc) / 2

                # find the closest avg soc for calendar aging
                cal_soc = np.asarray([0, 40, 90])
                closest_index = np.abs(cal_soc - avg_soc).argmin()
                closest = cal_soc[closest_index]

                if closest == 0:
                    cal_aging = self.calendar_aging_0 * dt / 8760
                elif closest == 40:
                    cal_aging = self.calendar_aging_40 * dt / 8760
                elif closest == 90:
                    cal_aging = self.calendar_aging_90 * dt / 8760
                else:
                    cal_aging = None
                    raise RuntimeError("Closest calendar aging SoC not recognised.")

                # calculate DoD of timestep
                dod = abs(new_soc - old_soc)

                # distinguish between high and low power charging according to input graph data
                if charging_power <= 22.0:
                    cycle_loss = dod * self.cycle_loss_11 / 2  # convert to equivalent full cycles, that's why divided by 2
                else:
                    cycle_loss = dod * self.cycle_loss_43 / 2  # convert to equivalent full cycles, that's why divided by 2

                # aggregate calendar and cyclic aging and append to degradation list
                degradation.append(cal_aging + cycle_loss)

                if (i % (8760 * 4 - 1) == 0):
                    soh_emp.append(1 - sum(degradation))
                # print(f"emp soh: {self.soh}")

        soh_emp = np.insert(soh_emp, 0, 1)
        return soh_emp
