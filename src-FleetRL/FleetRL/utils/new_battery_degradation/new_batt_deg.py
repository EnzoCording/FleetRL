from FleetRL.fleet_env.config.time_config import TimeConfig
class NewBatteryDegradation:
    def calculate_degradation(self, soc_list: list, charging_power: float, time_conf: TimeConfig, temp: float) -> float:
        '''
        :param soc_list: Historic values of SoC until now
        :time_conf: Time config file, necessary for time step length
        :return: Degradation of battery in kWh
        '''
        raise NotImplementedError("This is an abstract class.")