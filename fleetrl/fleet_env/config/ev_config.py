class EvConfig:
    """
    - The EV config sets some default parameters regarding the vehicles, charging infrastructure, efficiencies, etc.
    """
    def __init__(self):
        self.init_battery_cap: float = 60.0  # battery capacity in kWh
        self.obc_max_power: float = 100.0  # onboard charger max power in kW
        self.charging_eff: float = 0.91  # charging efficiency
        self.discharging_eff: float = 0.91  # discharging efficiency
        self.def_soc: float = 0.5  # default soc that is assumed in the beginning (for battery degradation)
        self.temperature: float = 25.0  # °C needed for battery degradation
        self.target_soc: float = 0.85  # Target soc when vehicle leaves
        self.target_soc_lunch = 0.65  # target soc after lunch break (for caretaker use-case)
        self.min_laxity: float = 2  # time left / time needed - minimum value for when a new episode starts
        self.fixed_markup: float = 10  # fixed cost added in €/MWh
        self.variable_multiplier: float = 1.5  # variable cost multiplier to (spot + fix)
        self.feed_in_deduction = 0.25  # 25%

