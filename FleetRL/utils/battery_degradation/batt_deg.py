from FleetRL.fleet_env.config.time_config import TimeConfig

class BatteryDegradation:
    def calculate_degradation(self, soc_list: list, charging_power: float, time_conf: TimeConfig, temp: float) -> float:
        """
        This is the parent class for degradation methods. Any new implemented method must follow this style of inputs
        and outputs. Then, the method can be used in FleetRL by changing one line of code in the import.

        The degradation methods in FleetRL are implemented such that degradation is calculated in real-time. In the
        step method of the environment class, the current SoH is calculated by SoH -= degradation

        :param soc_list: Historic values of SoC until now
        :param charging_power: Charging power in kW
        :param time_conf: Time config instance, necessary for time step length
        :param temp: Temperature to use in battery degradation in °C
        :return: Degradation of battery (unit-less, reduction of SoH, which is max. 1)
        """
        raise NotImplementedError("This is an abstract class.")
