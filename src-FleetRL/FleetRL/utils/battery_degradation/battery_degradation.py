class BatteryDegradation:

    def calculate_cycle_loss(self, old_soc: float, new_soc: float, charging_power: float) -> float:
        raise NotImplementedError("This is an abstract function")

    def calculate_calendar_aging_on_arrival(self, trip_time: float, old_soc: float, new_soc: float) -> float:
        raise NotImplementedError("This is an abstract function")

    def calculate_calendar_aging_while_parked(self, old_soc: float, new_soc: float, time_conf) -> float:
        raise NotImplementedError("This is an abstract function")
