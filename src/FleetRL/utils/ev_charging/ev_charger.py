import numpy as np
import pandas as pd

from FleetRL.fleet_env.config.ev_config import EvConfig
from FleetRL.fleet_env.config.score_config import ScoreConfig
from FleetRL.fleet_env.config.time_config import TimeConfig
from FleetRL.fleet_env.episode import Episode
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation


class EvCharger:

    def __init__(self, ev_conf: EvConfig):
        # Price analysis: https://www.bdew.de/media/documents/230215_BDEW-Strompreisanalyse_Februar_2023_15.02.2023.pdf
        # Average spot price for 2020 was ~3.2 ct/kWh and ~4 ct/kWh if looking at peak times only
        # --> 50% are fees --> spot price is multiplied by factor of 1.5 and offset by +1
        # this accounts for fees even when prices are negative or zero, but also scales with price levels
        self.spot_multiplier = ev_conf.variable_multiplier  # no unit
        self.spot_offset = ev_conf.fixed_markup / 1000  # from €/MWh to €/kWh

        # If energy is injected to the grid, it can be treated like solar feed-in from households
        # https://echtsolar.de/einspeiseverguetung/#t-1677761733663
        # Fees for handling of 25% are assumed
        self.handling_fees = ev_conf.feed_in_deduction  # %



    def charge(self,
               db: pd.DataFrame,
               num_cars: int,
               actions,
               episode: Episode,
               load_calculation: LoadCalculation,
               ev_conf: EvConfig,
               time_conf: TimeConfig,
               score_conf: ScoreConfig,
               print_updates: bool,
               target_soc: list):

        """
        :param db: The schedule database of the EVs
        :param num_cars: Number of cars in the model
        :param actions: Actions taken by the agent
        :param episode: Episode object with its parameters and functions
        :param load_calculation: Load calc object with its parameters and functions
        :param ev_conf: Config of the EVs
        :param time_conf: Time configuration
        :param score_conf: Score and penalty configuration
        :param print_updates: Bool whether to print statements or not (maybe lower fps)
        :param target_soc: target soc for each car
        :return: soc, next soc, the reward and the monetary value (cashflow)
        """

        # reset next_soc, cost and revenue
        episode.next_soc = []
        episode.charging_cost = 0
        episode.discharging_revenue = 0
        episode.total_charging_energy = 0

        # reset penalty counters
        invalid_action_penalty = 0
        overcharging_penalty = 0

        # reset energy values and log
        charge_log = np.ndarray(0)
        charging_energy = 0.0
        discharging_energy = 0.0

        charging_reward = 0.0
        discharging_reward = 0.0

        # go through the cars and calculate the actual deliverable power based on action and constraints
        for car in range(num_cars):

            # variable to check if car is plugged in or not
            there = db.loc[(db["ID"] == car) & (db["date"] == episode.time), "There"].values[0]

            # max possible power in kW depends on the onboard charger equipment and the charging station
            possible_power = min([ev_conf.obc_max_power, load_calculation.evse_max_power])
            # car is charging
            if actions[car] >= 0:
                # the charging energy depends on the maximum chargeable energy and the desired charging amount
                ev_total_energy_demand = (target_soc[car] - episode.soc[car]) * episode.battery_cap[car]  # total energy demand in kWh
                demanded_charge = possible_power * actions[car] * time_conf.dt  # demanded energy in kWh by the agent

                # if the agent wants to charge more than the battery can hold
                if demanded_charge * ev_conf.charging_eff > ev_total_energy_demand:
                    current_oc_pen = score_conf.penalty_overcharging * (demanded_charge - ev_total_energy_demand) ** 2
                    current_oc_pen = max(current_oc_pen, score_conf.clip_overcharging)
                    overcharging_penalty += current_oc_pen
                    if print_updates:
                        print(f"Overcharged, penalty of: {current_oc_pen}")

                # if the car is there, allocate charging energy to the battery in kWh
                if there == 1:
                    charging_energy = min(ev_total_energy_demand / ev_conf.charging_eff, demanded_charge)

                # the car is not there, no charging
                else:
                    charging_energy = 0
                    # if agent gives an action even if no car is there, give a small penalty
                    if np.abs(actions[car]) > 0.05:
                        current_inv_pen = score_conf.penalty_invalid_action * (actions[car] ** 2)
                        invalid_action_penalty += current_inv_pen
                        if print_updates:
                            print(f"Invalid action, penalty given: {round(current_inv_pen, 3)}.")

                # next soc is calculated based on charging energy
                episode.next_soc.append(episode.soc[car] + charging_energy * ev_conf.charging_eff / episode.battery_cap[car])

                # get pv energy and subtract from charging energy needed from the grid
                # assuming pv is equally distributed to the connected cars
                # try except because pv is sometimes deactivated
                try:
                    current_pv_energy = (db.loc[db["date"] == episode.time, "pv"].values[0]) * time_conf.dt  # in kWh
                except KeyError:
                    current_pv_energy = 0.0  # kWh
                connected_cars = db.loc[(db["date"] == episode.time), "There"].sum()
                # for the case that no car is connected, to avoid division by 0
                connected_cars = max(connected_cars, 1)
                # energy drawn from grid after deducting pv self-consumption
                grid_energy_demand = max(0, charging_energy - (current_pv_energy / connected_cars))  # kWh

                # get current spot price, div by 1000 to go from €/MWh to €/kWh
                current_spot = (db.loc[db["date"] == episode.time, "DELU"].values[0]) / 1000.0

                # calculate charging cost for this ev and add it to the total charging cost of the step
                episode.charging_cost += (grid_energy_demand * (current_spot + self.spot_offset) * self.spot_multiplier)

                # save the total charging energy in a variable
                episode.total_charging_energy += charging_energy

                charging_reward += (-1 * score_conf.price_multiplier
                                   * db.loc[db["date"]==episode.time, "price_reward_curve"].values[0] / 1000
                                   * grid_energy_demand)

            # car is discharging - v2g is currently modelled as energy arbitrage on the day ahead spot market
            elif actions[car] < 0:
                # check how much energy is left in the battery and how much discharge is desired
                ev_total_energy_left = -1 * episode.soc[car] * episode.battery_cap[car]  # amount of energy left in the battery in kWh
                demanded_discharge = possible_power * actions[car] * time_conf.dt  # demanded discharge in kWh by agent

                # energy discharge command bigger than what is left in the battery
                if (demanded_discharge * ev_conf.discharging_eff < ev_total_energy_left) and (there != 0):
                    current_oc_pen = score_conf.penalty_overcharging * (ev_total_energy_left - demanded_discharge) ** 2
                    overcharging_penalty += current_oc_pen
                    if print_updates:
                        print(f"Overcharged, penalty of: {round(current_oc_pen,3)}")

                # if the car is there get the actual discharging energy
                if there == 1:
                    discharging_energy = max(ev_total_energy_left, demanded_discharge)  # max because values are negative, kWh

                # car is not there, discharging energy is 0
                else:
                    discharging_energy = 0.0
                    # if discharge command is sent even if no car is there
                    if np.abs(actions[car]) > 0.05:
                        current_inv_pen = score_conf.penalty_invalid_action * (actions[car] ** 2)
                        invalid_action_penalty += current_inv_pen
                        if print_updates:
                            print(f"Invalid action, penalty given: {round(current_inv_pen, 3)}.")

                # calculate next soc
                # efficiency not taken into account here -> but you get out less (see below)
                episode.next_soc.append(episode.soc[car] + discharging_energy / episode.battery_cap[car])

                # Discharged energy renumerated at PV feed-in minus 30%
                # Efficiency taken into account here

                current_tariff = db.loc[db["date"] == episode.time, "tariff"].values[0]

                episode.discharging_revenue += (-1 * discharging_energy
                                                * ev_conf.discharging_eff
                                                * current_tariff / 1000
                                                * (1-self.handling_fees))  # €

                # save the total charging energy in a self variable
                episode.total_charging_energy += discharging_energy

                discharging_reward += (-1 * score_conf.price_multiplier
                                      * db.loc[db["date"]==episode.time, "tariff_reward_curve"].values[0] / 1000
                                      * discharging_energy)

            else:
                raise TypeError("The parsed action value was not recognised")

            # append total charging energy of the car to the charge log, used in post-processing
            charge_log = np.append(charge_log, charging_energy + discharging_energy)

            # Throw an error if SOC is actually negative
            if (np.round(episode.soc[car], 5) < 0) or (np.round(episode.soc[car], 5) > 1):
                print(f"SOC negative: {episode.soc[car]}"
                      f"Date: {episode.time}"
                      f"Action: {actions}"
                      f"Capacity: {episode.battery_cap[car]}")

            # Round off numeric inaccuracies (values in the range -1.0e-16 can happen otherwise and cause errors)
            np.clip(episode.soc, 0, 1)

        # calculate net cashflow based on cost and revenue
        cashflow = -1 * episode.charging_cost + episode.discharging_revenue

        # reward is a function of cashflow and penalties
        reward = charging_reward + discharging_reward + invalid_action_penalty + overcharging_penalty

        # return soc, next soc and the value of reward (remove the index)
        return episode.soc, episode.next_soc, float(reward), float(cashflow), charge_log
