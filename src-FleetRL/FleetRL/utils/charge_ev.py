def charge(self, action):
    # reset next_soc, cost and revenue
    self.next_soc = []
    charging_cost = 0
    discharging_revenue = 0
    self.total_charging_energy = 0
    invalid_action_penalty = 0

    # go through the cars and calculate the actual deliverable power based on action and constraints
    for car in range(self.cars):

        # possible power depends on the onboard charger equipment and the charging station
        possible_power = min([self.obc_max_power, self.evse_max_power])  # max possible charging power in kW

        # car is charging
        if action[car] >= 0:
            # the charging energy depends on the maximum chargeable energy and the desired charging amount
            ev_total_energy_demand = (1 - self.soc[car]) * self.battery_cap  # total energy demand in kWh
            demanded_charge = possible_power * action[car] * self.hours  # demanded energy in kWh

            # if the car is there
            if self.db.loc[(self.db["ID"] == car) & (self.db["date"] == self.time), "There"].values == 1:
                charging_energy = min([ev_total_energy_demand, demanded_charge])  # no overcharging or power violation

            # the car is not there
            else:
                charging_energy = 0
                if action[car] > 0:
                    print(f"Invalid action, penalty: {self.penalty_invalid_action}")
                    invalid_action_penalty += self.penalty_invalid_action

            # next soc is calculated based on charging energy
            # TODO: not all cars must have the same battery cap
            self.next_soc.append(self.soc[car] + charging_energy * self.charging_eff / self.battery_cap)

            # charging cost calculated based on spot price
            # TODO: add german taxes and grid fees
            charging_cost += (charging_energy *
                              self.spot_price.loc[self.spot_price["date"] == self.time, "DELU"]
                              ) / 1000
            # print(f"charging cost: {charging_cost.values[0]}")

            # save the total charging energy in a self variable
            self.total_charging_energy += charging_energy

        # car is discharging
        elif action[car] < 0:
            # check how much energy is left in the battery and how much discharge is desired
            ev_total_energy_left = -1 * self.soc[car] * self.battery_cap  # amount of energy left in the battery in kWh
            demanded_discharge = possible_power * action[car] * self.hours  # demanded discharge in kWh

            # if the car is there
            if self.db.loc[(self.db["ID"] == car) & (self.db["date"] == self.time), "There"].values == 1:
                discharging_energy = max(ev_total_energy_left, demanded_discharge)  # max because values are negative

            # car is not there
            else:
                discharging_energy = 0
                print(f"Invalid action, penalty: {self.penalty_invalid_action}")
                invalid_action_penalty += self.penalty_invalid_action

            # calculate next soc, which will get smaller
            self.next_soc.append(self.soc[car] + discharging_energy * self.discharging_eff / self.battery_cap)
            # TODO: variable prices, V2G?
            # TODO: FCR could be modelled by deciding to commit to not charging and then random soc flux
            discharging_revenue += (-1 * discharging_energy *
                                    self.spot_price.loc[self.spot_price["date"] == self.time, "DELU"]
                                    ) / 1000

            # print(f"discharging revenue: {discharging_revenue.values[0]}")

            # save the total charging energy in a self variable
            self.total_charging_energy += discharging_energy

        else:
            raise TypeError("The parsed action value was not recognised")

    # add reward based on cost and revenue
    charging_reward = -1 * charging_cost + discharging_revenue

    reward = charging_reward + invalid_action_penalty

    # return soc, next soc and the value of reward (remove the index)
    return self.soc, self.next_soc, float(reward)
