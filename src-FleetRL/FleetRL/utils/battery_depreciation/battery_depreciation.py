class BatteryDepreciation:

    def calculate(self, charging_time: float, initial_soc, temperature) -> float:
        raise NotImplementedError("This abstract function")
