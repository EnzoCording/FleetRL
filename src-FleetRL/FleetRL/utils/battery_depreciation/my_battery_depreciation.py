from FleetRL.utils.battery_depreciation.battery_depreciation import BatteryDepreciation


class MyBatteryDepreciation(BatteryDepreciation):

    def calculate(self, charging_time: float, initial_soc, temperature) -> float:
        # TODO
        return 0.1
