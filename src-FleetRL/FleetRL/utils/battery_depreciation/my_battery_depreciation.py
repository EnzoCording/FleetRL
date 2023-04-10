from FleetRL.utils.battery_depreciation.battery_dep_base import BatteryDepreciationBase


class MyBatteryDepreciation(BatteryDepreciationBase):

    def calculate(self, charging_time: float, initial_soc, temperature) -> float:
        x = super().calculate(charging_time, initial_soc, temperature)
        return x * 0.1
