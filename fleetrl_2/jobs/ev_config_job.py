from tomlchef.job import Job
from tomlchef.job import registered_job


class _Battery:
    def __init__(self,
                 battery_capacity: float = 60.0,
                 initial_state_of_health: float = 1.0,
                 on_board_charger_max_power: float = 100.0,
                 charging_efficiency: float = 0.91,
                 discharging_efficiency: float = 0.91,
                 default_soc: float = 0.5,
                 target_soc: float = 0.85):

        self.battery_capacity = battery_capacity
        self.initial_state_of_health = initial_state_of_health
        self.on_board_charger_max_power = on_board_charger_max_power
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency
        self.default_soc = default_soc
        self.target_soc = target_soc


@registered_job
class EvConfigJob(Job):

    def __init__(self,
                 battery: dict,
                 auxiliary: dict,
                 **kwargs):
        super().__init__(**kwargs)

        self.battery = _Battery(**battery)
        self.ambient_temperature: float = auxiliary['ambient_temperature']
        self.min_laxity: float = auxiliary['min_laxity']

    @staticmethod
    def get_toml_key() -> str:
        return "ev_config"

    def is_finished(self) -> bool:
        return True
