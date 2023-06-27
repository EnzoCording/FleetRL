import numpy as np
import pandas as pd

from FleetRL.fleet_env.config.ev_config import EvConfig
from FleetRL.fleet_env.config.score_config import ScoreConfig
from FleetRL.fleet_env.config.time_config import TimeConfig
from FleetRL.fleet_env.episode import Episode
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation


class EvCharger:

    def charge(self, db: pd.DataFrame, num_cars: int, actions, episode: Episode,
               load_calculation: LoadCalculation,
               ev_conf: EvConfig, time_conf: TimeConfig, score_conf: ScoreConfig, print_updates: bool, target_soc: float):

        """
        :param db: The schedule database of the EVs
        :param spot_price: Spot price information
        :param num_cars: Number of cars in the model
        :param actions: Actions taken by the agent
        :param episode: Episode object with its parameters and functions
        :param load_calculation: Load calc object with its parameters and functions
        :param soh: list that specifies the battery degradation of each vehicle
        :param ev_conf: Config of the EVs
        :param time_conf: Time configuration
        :param score_conf: Score and penalty configuration
        :param print_updates:
        :param target_soc:
        :return: soc, next soc, the reward and the monetary value (cashflow)
        """

        # reset next_soc, cost and revenue
        episode.next_soc = []
        episode.charging_cost = 0
        episode.discharging_revenue = 0
        episode.total_charging_energy = 0

        invalid_action_penalty = 0
        overcharging_penalty = 0

        charge_log = np.ndarray(0)
        charging_energy = 0.0
        discharging_energy = 0.0

        # go through the cars and calculate the actual deliverable power based on action and constraints
        for car in range(num_cars):

            # possible power depends on the onboard charger equipment and the charging station
            possible_power = min(
                [ev_conf.obc_max_power, load_calculation.evse_max_power])  # max possible charging power in kW

            # car is charging
            if actions[car] >= 0:
                # the charging energy depends on the maximum chargeable energy and the desired charging amount
                # SoH is accounted for in this equation as well
                ev_total_energy_demand = (target_soc - episode.soc[car] * episode.soh[car]) * ev_conf.battery_cap  # total energy demand in kWh
                demanded_charge = possible_power * actions[car] * time_conf.dt  # demanded energy in kWh

                if demanded_charge * ev_conf.charging_eff > ev_total_energy_demand:
                    current_oc_pen = score_conf.penalty_overcharging * (demanded_charge - ev_total_energy_demand) ** 2
                    overcharging_penalty += current_oc_pen
                    if print_updates:
                        print(f"Overcharged, penalty of: {current_oc_pen}")

                # if the car is there
                if db.loc[(db["ID"] == car) & (db["date"] == episode.time), "There"].values == 1:
                    charging_energy = min(ev_total_energy_demand / ev_conf.charging_eff, demanded_charge)

                # the car is not there
                else:
                    charging_energy = 0
                    if actions[car] > 0:
                        current_inv_pen = score_conf.penalty_invalid_action * (actions[car] ** 2)
                        invalid_action_penalty += current_inv_pen
                        if print_updates:
                            print(f"Invalid action, penalty given: {round(current_inv_pen, 3)}.")

                # next soc is calculated based on charging energy
                # TODO: not all cars must have the same battery cap
                episode.next_soc.append(episode.soc[car] * episode.soh[car]
                                        + charging_energy * ev_conf.charging_eff / ev_conf.battery_cap
                                        )

                # charging cost calculated based on spot price
                # TODO: add german taxes and grid fees
                # Divide by 1000 because we are in kWh

                # get pv and subtract from charging energy needed from the grid
                # assuming pv is equally distributed to the connected cars
                # try except because pv is sometimes deactivated
                try:
                    current_pv = db.loc[(db["ID"] == car) & (db["date"] == episode.time), "pv"].values[0]
                except KeyError:
                    current_pv = 0
                connected_cars = db.loc[(db["date"] == episode.time), "There"].sum()
                # for the case that no car is connected, to avoid division by 0
                connected_cars = max(connected_cars, 1)
                grid_demand = max(0, charging_energy - (current_pv / connected_cars))

                episode.charging_cost += (grid_demand *
                                          db.loc[db["date"] == episode.time, "DELU"].values[0]
                                          ) / 1000.0
                # print(f"charging cost: {charging_cost.values[0]}")

                # save the total charging energy in a self variable
                episode.total_charging_energy += charging_energy

            # car is discharging - v2g is currently modelled as energy arbitrage on the day ahead spot market
            elif actions[car] < 0:
                # check how much energy is left in the battery and how much discharge is desired
                ev_total_energy_left = -1 * episode.soc[car] * episode.soh[car] * ev_conf.battery_cap  # amount of energy left in the battery in kWh
                demanded_discharge = possible_power * actions[car] * time_conf.dt  # demanded discharge in kWh

                if demanded_discharge * ev_conf.discharging_eff < ev_total_energy_left:
                    current_oc_pen = score_conf.penalty_overcharging * (ev_total_energy_left - demanded_discharge) ** 2
                    overcharging_penalty += current_oc_pen
                    if print_updates:
                        print(f"Overcharged, penalty of: {round(current_oc_pen,3)}")

                # if the car is there
                if db.loc[(db["ID"] == car) & (db["date"] == episode.time), "There"].values == 1:
                    discharging_energy = max(ev_total_energy_left / ev_conf.discharging_eff, demanded_discharge)  # max because values are negative

                # car is not there
                else:
                    discharging_energy = 0
                    current_inv_pen = score_conf.penalty_invalid_action * (actions[car] ** 2)
                    invalid_action_penalty += current_inv_pen
                    if print_updates:
                        print(f"Invalid action, penalty given: {round(current_inv_pen, 3)}.")

                # calculate next soc, which will get smaller
                episode.next_soc.append(
                    episode.soc[car] * episode.soh[car]
                    + discharging_energy * ev_conf.discharging_eff / ev_conf.battery_cap
                )

                # Divide by 1000 because we are calculating in kWh
                episode.discharging_revenue += (-1 * discharging_energy *
                                                db.loc[db["date"] == episode.time, "DELU"].values[0]
                                                ) / 1000.0

                # print(f"discharging revenue: {discharging_revenue.values[0]}")

                # save the total charging energy in a self variable
                episode.total_charging_energy += discharging_energy

            else:
                raise TypeError("The parsed action value was not recognised")

            charge_log = np.append(charge_log, charging_energy + discharging_energy)

        # add reward based on cost and revenue
        cashflow = -1 * episode.charging_cost + episode.discharging_revenue

        reward = (score_conf.price_multiplier * cashflow) + invalid_action_penalty + overcharging_penalty

        # return soc, next soc and the value of reward (remove the index)
        return episode.soc, episode.next_soc, float(reward), float(cashflow), charge_log
