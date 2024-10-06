import numpy as np

from fleetrl.fleet_env.config.time_config import TimeConfig
from fleetrl.utils.battery_degradation.batt_deg import BatteryDegradation
from fleetrl_2.jobs.environment_creation_job import EpisodeParams


class EmpiricalDegradation(BatteryDegradation):

    def __init__(self, init_soh: float, num_cars: int):

        """
        Initialising the Degradation instance
        - http://queenbattery.com.cn/our-products/677-lg-e63-376v-63ah-li-po-li-polymer-battery-cell.html?search_query=lg+e63&results=1
        - read off the graphs in section 4: cycle and calendar aging
        :param init_soh: Initial state of health, assumed same for each EV
        :param num_cars: How many EVs are being optimized
        """

        self.cycle_loss_11 = 0.000125  # Cycle loss per full cycle (100% DoD discharge and charge) at 11 kW
        self.cycle_loss_22 = 0.000125  # Cycle loss per full cycle (100% DoD discharge and charge) at 11 kW
        self.cycle_loss_43 = 0.000167  # Cycle loss per full cycle (100% DoD discharge and charge) at 43 kW

        self.calendar_aging_0 = 0.0065  # Calendar aging per year if battery at 0% SoC
        self.calendar_aging_40 = 0.0293  # Calendar aging per year if battery at 40% SoC
        self.calendar_aging_90 = 0.065  # Calendar aging per year if battery at 90% SoC

        self.init_soh = init_soh
        self.num_cars = num_cars
        self.soh = np.ones(self.num_cars) * self.init_soh

    def calculate_degradation(self,
                              soc_log: list,
                              charging_power: float,
                              time_conf: EpisodeParams,
                              temp: float,
                              time_steps_per_hour: int) -> np.array:
        """
        Similar to non-linear SEI, the most recent event is taken, and the linear-based degradation is calculated.
        No rainflow counting, thus degradation is computed for each time step.

        - find out the most recent entries in the soc list
        - get old and new soc
        - get average soc
        - compute cycle and calendar based on avg soc and charging power

        :param time_steps_per_hour:
        :param soc_log: Historical log of SOC
        :param charging_power: EVSE power in kW
        :param time_conf: time config object
        :param temp: temperature
        :return: Degradation, float
        """

        # compute sorted soc list based on the log records of the episode so far
        # go from: t1:[soc_car1, soc_car2, ...], t2:[soc_car1, soc_car2,...]
        # to this: car 1: [soc_t1, soc_t2, ...], car 2: [soc_t1, soc_t2, ...]
        sorted_soc_list = []

        # range(len(soc_log[0])) gives the number of cars
        for j in range(len(soc_log[0])):
            # range(len(soc_log)) gives the number of time steps that the cars go through
            sorted_soc_list.append([soc_log[i][j] for i in range(len(soc_log))])


        # empty list, appends a degradation kWh value for each car
        degradation = []

        for i in range(len(sorted_soc_list)):

            # get old and new soc
            old_soc = sorted_soc_list[i][-2]
            new_soc = sorted_soc_list[i][-1]

            # compute average for calendar aging
            avg_soc = (old_soc + new_soc) / 2

            # find the closest avg soc for calendar aging
            cal_soc = np.asarray([0, 40, 90])
            closest_index = np.abs(cal_soc - avg_soc).argmin()
            closest = cal_soc[closest_index]

            if closest == 0:
                cal_aging = self.calendar_aging_0 * (1/time_steps_per_hour) / 8760
            elif closest == 40:
                cal_aging = self.calendar_aging_40 * (1/time_steps_per_hour) / 8760
            elif closest == 90:
                cal_aging = self.calendar_aging_90 * (1/time_steps_per_hour) / 8760
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

        self.soh -= degradation
        # print(f"emp soh: {self.soh}")

        return np.array(degradation)
