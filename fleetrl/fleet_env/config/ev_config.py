class EvConfig:
    """
    - The EV config sets some default parameters regarding the vehicles, charging infrastructure, efficiencies, etc.
    """

    def __init__(self, env_config):
        self.init_battery_cap: float = env_config.get("init_battery_cap", 60.0)  # battery capacity in kWh
        self.obc_max_power: float = env_config.get("obc_max_power", 100.0)  # onboard charger max power in kW
        self.charging_eff: float = env_config.get("charging_eff", 0.91)  # charging efficiency
        self.discharging_eff: float = env_config.get("discharging_eff", 0.91)  # discharging efficiency
        self.def_soc: float = env_config.get("def_soc", 0.5)  # default soc that is assumed (for battery degradation)
        self.temperature: float = env_config.get("temperature", 25.0)  # °C needed for battery degradation
        self.target_soc: float = env_config.get("target_soc", 0.85)  # Target soc when vehicle leaves
        self.target_soc_lunch = env_config.get("target_soc_lunch", 0.65)  # target soc after lunch break
        self.min_laxity: float = env_config.get("min_laxity", 2)  # time left / time needed - minimum value
        self.fixed_markup: float = env_config.get("fixed_markup", 10)  # fixed cost added in €/MWh
        self.variable_multiplier: float = env_config.get("variable_multiplier", 1.5)  # variable cost multiplier
        self.feed_in_deduction = env_config.get("feed_in_deduction", 0.25)  # 25% deducted for 3rd party services
