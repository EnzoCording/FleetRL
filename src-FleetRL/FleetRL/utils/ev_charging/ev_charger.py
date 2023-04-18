import numpy as np


class EvCharger:

    def charge(self, db, time, hours, battery_cap, charging_eff, discharging_eff, spot_price,
               soc, cars, actions, obc_max_power, evse_max_power, penalty_invalid_action):
        # reset next_soc, cost and revenue
        next_soc = []
        charging_cost = 0
        discharging_revenue = 0
        total_charging_energy = 0
        invalid_action_penalty = 0

        # go through the cars and calculate the actual deliverable power based on action and constraints
        for car in range(cars):

            # possible power depends on the onboard charger equipment and the charging station
            possible_power = min([obc_max_power, evse_max_power])  # max possible charging power in kW

            # car is charging
            if actions[car] >= 0:
                # the charging energy depends on the maximum chargeable energy and the desired charging amount
                ev_total_energy_demand = (1 - soc[car]) * battery_cap  # total energy demand in kWh
                demanded_charge = possible_power * actions[car] * hours  # demanded energy in kWh

                # if the car is there
                if db.loc[(db["ID"] == car) & (db["date"] == time), "There"].values == 1:
                    charging_energy = min(
                        [ev_total_energy_demand, demanded_charge])  # no overcharging or power violation

                # the car is not there
                else:
                    charging_energy = 0
                    if actions[car] > 0:
                        print(f"Invalid action, penalty: {penalty_invalid_action}")
                        invalid_action_penalty += penalty_invalid_action

                # next soc is calculated based on charging energy
                # TODO: not all cars must have the same battery cap
                next_soc.append(soc[car] + charging_energy * charging_eff / battery_cap)

                # charging cost calculated based on spot price
                # TODO: add german taxes and grid fees
                charging_cost += (charging_energy *
                                  spot_price.loc[spot_price["date"] == time, "DELU"]
                                  ) / 1000
                # print(f"charging cost: {charging_cost.values[0]}")

                # save the total charging energy in a self variable
                total_charging_energy += charging_energy

            # car is discharging
            elif actions[car] < 0:
                # check how much energy is left in the battery and how much discharge is desired
                ev_total_energy_left = -1 * soc[car] * battery_cap  # amount of energy left in the battery in kWh
                demanded_discharge = possible_power * actions[car] * hours  # demanded discharge in kWh

                # if the car is there
                if db.loc[(db["ID"] == car) & (db["date"] == time), "There"].values == 1:
                    discharging_energy = max(ev_total_energy_left,
                                             demanded_discharge)  # max because values are negative

                # car is not there
                else:
                    discharging_energy = 0
                    print(f"Invalid action, penalty: {penalty_invalid_action}")
                    invalid_action_penalty += penalty_invalid_action

                # calculate next soc, which will get smaller
                next_soc.append(soc[car] + discharging_energy * discharging_eff / battery_cap)
                # TODO: variable prices, V2G?
                # TODO: FCR could be modelled by deciding to commit to not charging and then random soc flux
                discharging_revenue += (-1 * discharging_energy *
                                        spot_price.loc[spot_price["date"] == time, "DELU"]
                                        ) / 1000

                # print(f"discharging revenue: {discharging_revenue.values[0]}")

                # save the total charging energy in a self variable
                total_charging_energy += discharging_energy

            else:
                raise TypeError("The parsed action value was not recognised")

        # add reward based on cost and revenue
        charging_reward = -1 * charging_cost + discharging_revenue

        reward = charging_reward + invalid_action_penalty

        # return soc, next soc and the value of reward (remove the index)
        return soc, next_soc, float(reward)
