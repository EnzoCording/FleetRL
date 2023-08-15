# import rainflow as rf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# data = pd.read_csv("soc_annual.csv")
# soc_lin_lmd_arb = pd.read_csv("/home/enzo/Downloads/soc_lin_arb_updated.csv")
# soc_rl_lmd_arb = pd.read_csv("/home/enzo/Downloads/soc_rl_arb_updated.csv")
# num_cars = 1
#
# # Source: Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment
# # https://ieeexplore.ieee.org/document/7488267
#
# # initial state of health of the battery
# init_soh = 1.0
# soh = np.ones(1) * init_soh
#
# # battery life, according to the paper notation
# l = np.ones(1) - soh
#
# # non-linear degradation model
# alpha_sei = 5.75E-2
# beta_sei = 121
#
# # DoD stress model
# kd1 = 1.4E5
# kd2 = -5.01E-1
# kd3 = -1.23E5
#
# # SoC stress model
# k_sigma = 1.04
# sigma_ref = 0.5
#
# # Temperature stress model
# k_temp = 6.93E-2
# temp_ref = 25  # °C
#
# # Calendar aging model
# k_dt = 4.14E-10  # 1/s -> per second
#
# def stress_dod(dod): return (kd1 * (dod ** kd2) + kd3) ** -1
#
# def stress_soc(soc): return np.e ** (k_sigma * (soc - sigma_ref))
#
# def stress_temp(temp): return np.e ** (k_temp * (temp - temp_ref)
#                                              * ((temp_ref + 273.15) / (temp + 273.15)))
#
# def stress_time(t): return k_dt * t
#
# def deg_rate_cycle(dod, avg_soc, temp): return (stress_dod(dod)
#                                                       * stress_soc(avg_soc)
#                                                       * stress_temp(temp))
#
# def deg_rate_calendar(t, avg_soc, temp): return (stress_time(t)
#                                                        * stress_soc(avg_soc)
#                                                        * stress_temp(temp))
#
# def deg_rate_total(dod, avg_soc, temp, t): return (deg_rate_cycle(dod, avg_soc, temp)
#                                                          + deg_rate_calendar(t, avg_soc, temp))
#
# def l_with_sei(fd): return (1-alpha_sei*np.e**(-beta_sei * fd)-(1-alpha_sei)*np.e**(-1*fd))
#
#
# def calculate_degradation(soc_log = data, charging_power: float=11.0, temp: float=25.0) -> np.array:
#
#     # compute sorted soc list based on the log records of the episode so far
#     # go from: t1:[soc_car1, soc_car2, ...], t2:[soc_car1, soc_car2,...]
#     # to this: car 1: [soc_t1, soc_t2, ...], car 2: [soc_t1, soc_t2, ...]
#
#     sorted_soc_list = soc_log
#
#     # this is 0 in the beginning and then gets updated with the new degradation due to the current time step
#     degradation = np.zeros(len(sorted_soc_list))
#
#     # calculate rainflow list and store it somewhere
#     # check its length and see if it increased by one
#     # if it increased, calculate with the previous entry, otherwise pass
#     # I need the 0.5 / 1.0, the start and end, the average, the DoD
#
#     rainflow_df = pd.DataFrame(columns=['Range', 'Mean', 'Count', 'Start', 'End'])
#
#     for rng, mean, count, i_start, i_end in rf.extract_cycles(np.tile(sorted_soc_list, 1)):
#         new_row = pd.DataFrame(
#             {'Range': [rng], 'Mean': [mean], 'Count': [count], 'Start': [i_start], 'End': [i_end]})
#         rainflow_df = pd.concat([rainflow_df, new_row], ignore_index=True)
#
#     L_sei = []
#
#     # len(sorted_soc_list) gives the number of cars
#     for i in range(1,11):
#
#         rnflow_data = rainflow_df.loc[rainflow_df.index.repeat((i) * 8760 * 4 / len(sorted_soc_list))]
#
#         # if len(rnflow_data) == 0:
#         #     continue
#
#         # dod is equal to the range
#         dod = rnflow_data["Range"]
#
#         # average soc is equal to the mean
#         avg_soc = rnflow_data["Mean"]
#
#         if len(avg_soc) == 0:
#             mean_cal = soc_log.mean()
#         else:
#             mean_cal = avg_soc.mean()
#
#         # severity is equal to count: either 0.5 or 1.0
#         degradation_severity = rnflow_data["Count"]
#
#         # deg_rate_total becomes negative for DoD > 1
#         # the two checks below count how many times dod is adjusted and in severe cases stops the code
#
#         # half or full cycle, max of 1
#         effective_dod = np.clip(dod * degradation_severity, 0, 1)
#
#         # time of the cycle
#         t = (rnflow_data["End"] - rnflow_data["Start"]) * 0.25 * 3600
#         t_cal = np.max(rnflow_data["End"])*0.25*3600
#
#         f_cyc = deg_rate_cycle(effective_dod, avg_soc, temp)
#         f_cal = deg_rate_calendar(t_cal*i, avg_soc=mean_cal, temp=temp)
#
#         # check if new battery, otherwise ignore sei film formation
#         if init_soh == 1.0:
#             fd = f_cyc.sum() + f_cal
#             new_l = l_with_sei(fd)
#
#             # check if l is negative, then something is wrong
#             if new_l < 0:
#                 raise TypeError("Life degradation is negative")
#
#         # if battery used, sei film formation is done and can be ignored
#         else:
#             fd[0] += deg_rate_total(effective_dod, avg_soc, temp, t)
#
#         # calculate degradation based on the change of l
#         degradation[0] = new_l - l[0]
#
#         # update lifetime variable
#         l[0] = new_l
#
#         soh[0] -= degradation[0]
#
#         L_sei = np.append(L_sei, l)
#
#     # check that the adding up of degradation is equivalent to the newest lifetime value calculated
#     if abs(soh[0] - (1 - l[0])) > 0.0001:
#         raise RuntimeError("Degradation calculation is not correct")
#
#     # print(f"sei soh: {soh}")
#
#     print(L_sei)
#     print(soh)
#
#     return np.argmax(L_sei >= 0.2) + 1, 1 - L_sei
#
#
# data = pd.read_csv("soc_annual.csv")
# data = np.array(data["soc"].values)
# a = calculate_degradation(data)
#
# cycle_loss_11 = 0.000125  # Cycle loss per full cycle (100% DoD discharge and charge) at 11 kW
# cycle_loss_22 = 0.000125  # Cycle loss per full cycle (100% DoD discharge and charge) at 11 kW
# cycle_loss_43 = 0.000167  # Cycle loss per full cycle (100% DoD discharge and charge) at 43 kW
#
# calendar_aging_0 = 0.0065  # Calendar aging per year if battery at 0% SoC
# calendar_aging_40 = 0.0293  # Calendar aging per year if battery at 40% SoC
# calendar_aging_90 = 0.065  # Calendar aging per year if battery at 90% SoC
#
# sorted_soc_list = data
# charging_power = 11
# dt = 0.25
# soh_emp = []
# degradation = []
#
# for j in range(1, 11):
#
#     for i in range(1, len(data)):
#
#         old_soc = sorted_soc_list[i - 1]
#         new_soc = sorted_soc_list[i]
#
#         # compute average for calendar aging
#         avg_soc = (old_soc + new_soc) / 2
#
#         # find the closest avg soc for calendar aging
#         cal_soc = np.asarray([0, 40, 90])
#         closest_index = np.abs(cal_soc - avg_soc).argmin()
#         closest = cal_soc[closest_index]
#
#         if closest == 0:
#             cal_aging = calendar_aging_0 * dt / 8760
#         elif closest == 40:
#             cal_aging = calendar_aging_40 * dt / 8760
#         elif closest == 90:
#             cal_aging = calendar_aging_90 * dt / 8760
#         else:
#             cal_aging = None
#             raise RuntimeError("Closest calendar aging SoC not recognised.")
#
#         # calculate DoD of timestep
#         dod = abs(new_soc - old_soc)
#
#         # distinguish between high and low power charging according to input graph data
#         if charging_power <= 22.0:
#             cycle_loss = dod * cycle_loss_11 / 2  # convert to equivalent full cycles, that's why divided by 2
#         else:
#             cycle_loss = dod * cycle_loss_43 / 2  # convert to equivalent full cycles, that's why divided by 2
#
#         # aggregate calendar and cyclic aging and append to degradation list
#         degradation.append(cal_aging + cycle_loss)
#
#         if (i % (8760 * 4 - 1) == 0):
#             soh_emp.append(1 - sum(degradation))
#         # print(f"emp soh: {self.soh}")
#
# soh_emp = np.insert(soh_emp, 0, 1)
# a = (a[0], np.insert(a[1], 0, 1))
# plt.plot(a[1])
# plt.plot(soh_emp)
# plt.grid(True)
# plt.xlim([0,10])
# plt.ylim([0.8, 1])
# plt.legend(["SEI formation + Cycle counting", "Empirical linear degradation"])
# plt.xticks(range(11))
# plt.yticks([0.8,0.85,0.9,0.95,1.0])
# plt.savefig("sei_emp_10.pdf")
# plt.show()

import pandas as pd
from FleetRL.utils.battery_degradation.compare_methods import Comparison

soc_lin_ct_arb = pd.read_csv("./v2/soc_lin_ct_arb_v2_cleaned.csv")
soc_rl_ct_arb = pd.read_csv("./v2/soc_rl_ct_arb_v2_cleaned.csv")
soc_lin_ct_real = pd.read_csv("./v2/soc_lin_ct_real_v2_cleaned.csv")
soc_rl_ct_real = pd.read_csv("./v2/soc_rl_ct_real_v2_cleaned.csv")

soc_lin_lmd_arb = pd.read_csv("./v2/soc_lin_lmd_arb_v2_cleaned.csv")
soc_rl_lmd_arb = pd.read_csv("./v2/soc_rl_lmd_arb_v2_cleaned.csv")
soc_lin_lmd_real = pd.read_csv("./v2/soc_lin_lmd_real_v2_cleaned.csv")
soc_rl_lmd_real = pd.read_csv("./v2/soc_rl_lmd_real_v2_cleaned.csv")

soc_lin_ut_arb = pd.read_csv("./v2/soc_lin_ut_arb_v2_cleaned.csv")
soc_rl_ut_arb = pd.read_csv("./v2/soc_rl_ut_arb_v2_cleaned.csv")
soc_lin_ut_real = pd.read_csv("./v2/soc_lin_ut_real_v2_cleaned.csv")
soc_rl_ut_real = pd.read_csv("./v2/soc_rl_ut_real_v2_cleaned.csv")

soc_dfs = [soc_rl_lmd_arb, soc_lin_lmd_arb, soc_rl_lmd_real, soc_lin_lmd_real]
m=0
for df in soc_dfs:
    for i in range(len(df)):
        if i == (len(df) - 1):
            continue
        if (df.loc[i, "soc"] > 0) and (df.loc[i+1, "soc"] == 0):
            df.loc[i, "soc"] = 0.85
            df.loc[i-1, "soc"] = 0.8
            df.loc[i-2, "soc"] = 0.75
            df.loc[i-3, "soc"] = 0.7
            df.loc[i-4, "soc"] = 0.65
            df.loc[i-5, "soc"] = 0.5
            df.loc[i-6, "soc"] = 0.45
            df.loc[i-7, "soc"] = 0.4
            df.loc[i-8, "soc"] = 0.35
            df.loc[i-9, "soc"] = 0.3

    soc_dfs[m] = df
    m+=1

comp = Comparison()
comp.compare_methods(soc_dfs, save=True)
