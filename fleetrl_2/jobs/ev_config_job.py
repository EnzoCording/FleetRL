from tidysci.task import Task
from tidysci.task import register


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


@register(alias=True)
class EvConfigJob(Task):

    def __init__(self,
                 battery: dict,
                 auxiliary: dict,
                 _dir_root: str,
                 rng_seed: int):

        super().__init__(_dir_root, rng_seed)

        self.battery = _Battery(**battery)
        self.ambient_temperature: float = auxiliary['ambient_temperature']
        self.min_laxity: float = auxiliary['min_laxity']

    def is_finished(self) -> bool:
        return True
