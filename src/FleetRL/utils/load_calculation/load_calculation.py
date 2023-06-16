# import pandapower as pp
from enum import Enum


class CompanyType(Enum):
    Delivery = 1
    Caretaker = 2
    Utility = 3


class LoadCalculation:

    @staticmethod
    def _import_company(company_type: CompanyType):
        grid_connection: float  # max grid connection in kW
        evse_power: float  # charger capacity in kW

        if company_type == CompanyType.Delivery:
            grid_connection = 100
            evse_power = 11

        elif company_type == CompanyType.Utility:
            grid_connection = 100  # max grid connection in kW
            evse_power = 22  # charger cap in kW

        elif company_type == CompanyType.Caretaker:
            grid_connection = 120  # max grid in kW
            evse_power = 7.4  # charger cap in kW

        else:
            grid_connection = 20
            evse_power = 3.7
            print("WARN: Company name not found. Default values loaded: grid: 20 kW, evse: 3.7 kW.")

        return grid_connection, evse_power

    def __init__(self, company_type: CompanyType):
        # setting parameters of the company site
        self.company_type = company_type

        # TODO: max power could change, I even have that info in the schedule
        # Grid connection: grid connection point max capacity in kW
        # EVSE (ev supply equipment aka charger) max power in kW
        self.grid_connection, self.evse_max_power = LoadCalculation._import_company(self.company_type)

    def build_grid(self):
        # TODO building the grid, figure out later
        pass

    def check_violation(self, actions: list[float], building_load: float, pv: float) -> (bool, float):
        # grid connection - building load - total ev + pv >= 0
        # TODO: double check that this makes sense from a component point of view
        capacity_left = self.grid_connection - building_load - sum(actions) * self.evse_max_power + pv
        return capacity_left >= 0, capacity_left
