# import pandapower as pp
from enum import Enum

class CompanyType(Enum):
    Delivery = 1
    Caretaker = 2
    Utility = 3

class LoadCalculation:

    @staticmethod
    def _import_company(company_type: CompanyType, max_load: float, num_cars: int):
        grid_connection: float  # max grid connection in kW
        evse_power: float  # charger capacity in kW

        if company_type == CompanyType.Delivery:
            evse_power = 11
            grid_connection = max(max_load*1.1, max_load + 0.5*num_cars*evse_power)
            batt_cap = 60  # evito

        elif company_type == CompanyType.Utility:
            evse_power = 22  # charger cap in kW
            grid_connection = max(max_load*1.1, max_load + 0.5*num_cars*evse_power)
            if num_cars > 1:
                grid_connection = 1000
            batt_cap = 50  # e berlingo

        elif company_type == CompanyType.Caretaker:
            evse_power = 4.6  # charger cap in kW
            grid_connection = max(max_load*1.1, max_load + 0.5*num_cars*evse_power)
            batt_cap = 16.7  # smart eq

        else:
            grid_connection = 200
            evse_power = 3.7
            batt_cap = 35
            print("WARN: Company name not found. Default values loaded.")

        return grid_connection, evse_power, batt_cap

    def __init__(self, company_type: CompanyType, max_load: float, num_cars: int):
        # setting parameters of the company site
        self.company_type = company_type
        self.max_load = max_load
        self.num_cars = num_cars

        # TODO: max power could change, I even have that info in the schedule
        # Grid connection: grid connection point max capacity in kW
        # EVSE (ev supply equipment aka charger) max power in kW
        self.grid_connection, self.evse_max_power, self.batt_cap = LoadCalculation._import_company(self.company_type,
                                                                                                   self.max_load,
                                                                                                   self.num_cars)

    def check_violation(self, actions: list[float], there: list[int], building_load: float, pv: float) -> (bool, float):
        # grid connection - building load - total ev + pv >= 0
        # TODO: double check that this makes sense from a component point of view
        capacity_left = min(self.grid_connection - building_load - sum(actions) * self.evse_max_power + pv, 0.0)
        return capacity_left < 0, capacity_left
