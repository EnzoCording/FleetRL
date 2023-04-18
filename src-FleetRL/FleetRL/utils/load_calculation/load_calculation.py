# import pandapower as pp

class LoadCalculation:

    def import_company(self):
        grid_connection: float  # max grid connection in kW
        evse_power: float  # charger capacity in kW

        if self.company_name == "delivery":
            grid_connection = 100
            evse_power = 11

        elif self.company_name == "utility":
            grid_connection = 100  # max grid connection in kW
            evse_power = 22  # charger cap in kW

        elif self.company_name == "caretaker":
            grid_connection = 30  # max grid in kW
            evse_power = 7.4  # charger cap in kW

        else:
            grid_connection = 20
            evse_power = 3.7
            print("WARN: Company name not found. Default values loaded: grid: 20 kW, evse: 3.7 kW.")

        return grid_connection, evse_power

    def __init__(self, company_name: str):
        # setting parameters of the company site
        self.company_name = company_name    # "delivery", "caretaker", "utility"

        # TODO: max power could change, I even have that info in the schedule
        # Grid connection: grid connection point max capacity in kW
        # EVSE (ev supply equipment aka charger) max power in kW
        self.grid_connection, self.evse_max_power = self.import_company()

    def build_grid(self):
        # TODO building the grid, figure out later
        pass

    def check_violation(self, building_load: float,
                        action: list[float], pv: float):
        # grid connection - building load - total ev + pv > 0
        # TODO: double check that this makes sense from a component point of view
        capacity_left = self.grid_connection - building_load - sum(action) * self.evse_max_power + pv
        return capacity_left >= 0
