from FleetRL.utils.battery_depreciation.battery_dep_base import BatteryDepreciationBase


class MyBatteryDepreciation(BatteryDepreciationBase):

    def calculate(self, charging_time: float, initial_soc, temperature) -> float:
        return 0.1
