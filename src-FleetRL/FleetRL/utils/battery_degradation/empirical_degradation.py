import numpy as np

from FleetRL.utils.battery_degradation.battery_degradation import BatteryDegradation


class EmpiricalDegradation(BatteryDegradation):

    def __init__(self):
        self.cycle_loss_11 = 0.000125  # Cycle loss per full cycle (100% DoD discharge and charge) at 11 kW
        self.cycle_loss_22 = 0.000125  # Cycle loss per full cycle (100% DoD discharge and charge) at 11 kW
        self.cycle_loss_43 = 0.000167  # Cycle loss per full cycle (100% DoD discharge and charge) at 43 kW

        self.calendar_aging_0 = 0.0065  # Calendar aging per year if battery at 0% SoC
        self.calendar_aging_40 = 0.0293  # Calendar aging per year if battery at 40% SoC
        self.calendar_aging_90 = 0.065  # Calendar aging per year if battery at 90% SoC

    def calculate_cycle_loss(self, old_soc, new_soc, charging_power) -> float:

        dod = abs(new_soc - old_soc)

        if charging_power <= 22:
            cycle_loss = dod * self.cycle_loss_11 / 2  # convert to equivalent full cycles, that's why divided by 2
        else:
            cycle_loss = dod * self.cycle_loss_43 / 2  # convert to equivalent full cycles, that's why divided by 2

        return cycle_loss

    def calculate_calendar_aging_on_arrival(self, trip_time: float, old_soc, new_soc) -> np.array:

        avg_soc = (old_soc + new_soc) / 2
        cal_soc = np.asarray([0, 40, 90])
        closest_index = np.abs(cal_soc - avg_soc).argmin()
        closest = cal_soc[closest_index]

        if closest == 0:
            cal_aging = self.calendar_aging_0 * trip_time / 8760
        elif closest == 40:
            cal_aging = self.calendar_aging_40 * trip_time / 8760
        elif closest == 90:
            cal_aging = self.calendar_aging_90 * trip_time / 8760
        else:
            cal_aging = None
            raise RuntimeError("Closest calendar aging SoC not recognised.")

        return cal_aging

    def calculate_calendar_aging_while_parked(self, old_soc: float, new_soc: float, time_conf) -> float:

        avg_soc = (old_soc + new_soc) / 2
        cal_soc = np.asarray([0, 40, 90])
        closest_index = np.abs(cal_soc - avg_soc).argmin()
        closest = cal_soc[closest_index]

        if closest == 0:
            cal_aging = self.calendar_aging_0 * time_conf.dt / 8760
        elif closest == 40:
            cal_aging = self.calendar_aging_40 * time_conf.dt / 8760
        elif closest == 90:
            cal_aging = self.calendar_aging_90 * time_conf.dt / 8760
        else:
            cal_aging = None
            raise RuntimeError("Closest calendar aging SoC not recognised.")

        return cal_aging
