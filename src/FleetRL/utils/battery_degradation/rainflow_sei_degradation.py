import rainflow as rf

from FleetRL.utils.battery_degradation.battery_degradation import BatteryDegradation

class RainFlowSei(BatteryDegradation):
    def print_rainflow(self, soc_list, cars):
        for car in range(cars-10):
            signal = [soc_list[i][car] for i in range(len(soc_list))]
            for rng, mean, count, i_start, i_end in rf.extract_cycles(signal):
                print(rng, mean, count, i_start, i_end)

    def calculate_cycle_loss(self, old_soc: float, new_soc: float, charging_power: float, soc_list, cars) -> float:
        for car in range(cars):
            signal = [soc_list[i][car] for i in range(len(soc_list))]
            for rng, mean, count, i_start, i_end in rf.extract_cycles(signal):
                print(rng, mean, count, i_start, i_end)