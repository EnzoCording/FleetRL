from enum import Enum

from fleetrl_2.jobs.ev_config_job import EvConfigJob
from fleetrl_2.jobs.schedule_parameters.schedule_parameters import ScheduleParameters
from fleetrl_2.jobs.site_parameters_job import SiteParametersJob


class CompanyType(Enum):
    Delivery = 1
    Caretaker = 2
    Utility = 3
    Custom = 4

class LoadCalculation:
    """
    The Load Calculation class sets grid connection, and calculates load overloading
    """

    @staticmethod
    def _import_company(env_config: dict,
                        company_type: CompanyType,
                        max_load: float,
                        num_cars: int):
        """
        The grid connection is calculated based on the building load, number of EVs and the EVSE kW power.
        The grid connection is either 1.1 * max_load, or such that a simultaneous charging of 50% of EVs would overload
        the trafo at max building load.

        :param company_type: Utility, Last-mile delivery, or caretaker, custom
        :param max_load: The max building load of the use-case in kW
        :param num_cars: Number of EVs
        :return: Grid connection in kW, EVSE power in kW, batt_cap in kWh
        """
        grid_connection: float  # max grid connection in kW
        evse_power: float  # charger capacity in kW

        if company_type == CompanyType.Delivery:
            evse_power = 11
            grid_connection = max(max_load*1.1, max_load + 0.5*num_cars*evse_power)
            batt_cap = 60  # e-vito

        elif company_type == CompanyType.Utility:
            evse_power = 22  # charger cap in kW
            grid_connection = max(max_load*1.1, max_load + 0.5*num_cars*evse_power)
            if num_cars > 1:
                grid_connection = 1000
            batt_cap = 50  # e-berlingo

        elif company_type == CompanyType.Caretaker:
            evse_power = 4.6  # charger cap in kW
            grid_connection = max(max_load*1.1, max_load + 0.5*num_cars*evse_power)
            batt_cap = 16.7  # smart eq

        elif company_type == CompanyType.Custom:
            evse_power = env_config.get("custom_ev_charger_power_in_kw", 120)
            grid_connection = env_config.get("custom_grid_connection_in_kw", 500)
            batt_cap = env_config.get("custom_ev_battery_size_in_kwh", 60)

        else:
            grid_connection = 200
            evse_power = 3.7
            batt_cap = 35
            print("WARN: Company name not found. Default values loaded.")

        return grid_connection, evse_power, batt_cap

    def __init__(self,
                 ev_config: EvConfigJob,
                 battery_capacity: float,
                 max_grid_connection: float,
                 max_load: float,
                 num_cars: int,
                 charging_power: float):
        """
        Initialize the Load Calculation module

        :param ev_config: Stores ev related data
        :param site_parameters: Stores site related data
        :param max_load: Max load in the dataset
        :param num_cars: Number of cars to be optimized
        :param schedule_parameters: The charger capacity is stored in here
        """

        # setting parameters of the company site
        self.max_load = max_load
        self.num_cars = num_cars
        self.grid_connection = max_grid_connection
        self.evse_max_power = charging_power
        self.batt_cap = battery_capacity

    def check_violation(self, actions: list[float], there: list[int], building_load: float, pv: float) -> (bool, float):
        """

        :param actions: Actions list, action for each EV [-1,1]
        :param there: Flag is EV is plugged in or not [0;1]
        :param building_load: Current building load in kW
        :param pv: Current PV in kW
        :return:
        """
        # check if overloaded and by how much, PV is subtracted
        overload_amount = abs(min(self.grid_connection - building_load - sum(actions) * self.evse_max_power + pv, 0.0))
        return overload_amount > 0, overload_amount
