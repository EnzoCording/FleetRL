# import pandapower as pp

def import_company(company_name):
    if company_name == "delivery":
        grid_connection = 100  # max grid connection in kW
        evse_power = 11  # charger capacity in kW

    elif company_name == "utility":
        grid_connection = 100  # max grid connection in kW
        evse_power = 22  # charger cap in kW

    elif company_name == "caretaker":
        grid_connection = 30  # max grid in kW
        evse_power = 7.4  # charger cap in kW

    else:
        grid_connection = 20  # default value in kW
        evse_power = 3.7  # default value in kW
        print("Company name not found. Default values loaded: grid: 20 kW, evse: 3.7 kW.")

    return grid_connection, evse_power


def build_grid(self):
    # building the grid, figure out later
    pass


def check_violation(self, action):
    # grid connection - building load - total ev + pv > 0
    # TODO: double check that this makes sense from a component point of view
    capacity_left = self.grid_connection - self.building_load - sum(action) * self.evse_max_power + self.pv
    return capacity_left >= 0
