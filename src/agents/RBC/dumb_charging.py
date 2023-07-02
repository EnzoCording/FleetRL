import FleetRL
from FleetRL.fleet_env.fleet_environment import FleetEnv
from FleetRL.utils.battery_degradation.new_rainflow_sei_degradation import NewRainflowSeiDegradation
from FleetRL.utils.battery_degradation.new_empirical_degradation import NewEmpiricalDegradation

import pandas as pd
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import rainflow as rf

env = FleetEnv()
env.reset()
new_run = False
steps = 8760 * env.time_conf.time_steps_per_hour

if new_run:
    for i in range(steps):
        env.step(np.ones(env.num_cars))

    soc_log = []
    soh_log = []
    soh_2 = []

    soc_annual = pd.DataFrame(soc_log[0:8760*4][0])
    soc_annual.to_csv("soc_annual.csv")

# soc_10_years = pd.DataFrame()
# for i in range(10):
#     soc_10_years = pd.concat([soc_10_years, soc_annual], axis=0, ignore_index=True)
#
# soc_10_years.to_csv("soc_10_years.csv")
#
# soc_10_log = []
# deg = 0
# deg_emp = 0
# sei_deg = NewRainflowSeiDegradation(env.initial_soh, env.num_cars)
# emp_deg = NewEmpiricalDegradation(env.initial_soh, env.num_cars)
#
# for i in soc_10_years.index:
#     deg += sei_deg.calculate_degradation(soc_10_years[0:i],env.load_calculation.evse_max_power, env.time_conf, env.ev_conf.temperature)
#     deg_emp += emp_deg.calculate_degradation(soc_10_years[0:i], env.load_calculation.evse_max_power, env.time_conf, env.ev_conf.temperature)
#     soc_10_log.append([1 - deg, 1 - deg_emp])
#
# pd.DataFrame(soc_10_log).to_csv("10_years_log.csv")
#
# plt.plot(soc_10_log[0])
# plt.plot(soc_10_log[1])
# plt.show()

def MyFun(SOC_profile: np.array):
    # parameters
    # non-linear degradation
    a_sei = 5.75e-2
    b_sei = 121
    # DoD stress
    k_d1 = 1.4e5
    k_d2 = -5.01e-1
    k_d3 = -1.23e5
    # SoC stress
    k_s = 1.04
    s_ref = 0.5
    # temperature stress
    k_T = 6.93e-2
    T_ref = 25  # degC
    # calenar ageing
    k_t = 4.14e-10  # 1/second

    # functions
    funct_S_d = lambda d: (k_d1 * d ** k_d2 + k_d3) ** (-1)  # DoD degradation
    funct_S_s = lambda s: np.exp(k_s * (s - s_ref))  # SOC degradation
    funct_S_T = lambda T: np.exp(k_T * (T - T_ref) * T_ref / T)  # Temperature degradation
    funct_S_t = lambda t: t * k_t  # time degradation

    funct_f_cyc_i = lambda d, s, T: funct_S_d(d) * funct_S_s(s) * funct_S_T(T)  # cyclic ageing
    funct_f_cal = lambda s, t, T: funct_S_s(s) * funct_S_t(t) * funct_S_T(T)  # calendar ageing

    L = np.array([])
    L_sei = np.array([])

    rainflow = pd.DataFrame(columns=['Range', 'Mean', 'Count', 'Start', 'End'])

    for rng, mean, count, i_start, i_end in rf.extract_cycles(np.tile(SOC_profile, 1)):
        new_row = pd.DataFrame({'Range': [rng], 'Mean': [mean], 'Count': [count], 'Start': [i_start], 'End': [i_end]})
        rainflow = pd.concat([rainflow, new_row], ignore_index=True)

    for i in range(1, 11):
        rnflow_data = rainflow.loc[rainflow.index.repeat((i - 1) * 8760 * 4 / len(SOC_profile))]

        rf.count_cycles(SOC_profile)

        DoD = rnflow_data['Range']
        SOC = rnflow_data['Mean']
        f_cyc = funct_f_cyc_i(DoD, SOC, T_ref) * rnflow_data[
            'Count']  # I multiply the weight of the cycle by the degradation of that cycle
        SOC_avg = SOC_profile.mean()
        f_cal = funct_f_cal(SOC_avg, 3600 * 8760 * i, T_ref)
        f_d = f_cyc.sum() + f_cal
        L = np.append(L, [1 - np.exp(-f_d)])
        L_sei = np.append(L_sei, [1 - a_sei * np.exp(-b_sei * f_d) - (1 - a_sei) * np.exp(-f_d)])

    return np.argmax(
        L_sei >= 0.2) + 1, 1 - L_sei  # the first is the cyclelife of the battery for a given SOC profile of 1 year, supposing that the battery periodically implements that strategy
    # the secon one is the SOH of the battery every year


data = pd.read_csv("soc_annual.csv")
data = np.array(data["soc"].values)
a = MyFun(data)

cycle_loss_11 = 0.000125  # Cycle loss per full cycle (100% DoD discharge and charge) at 11 kW
cycle_loss_22 = 0.000125  # Cycle loss per full cycle (100% DoD discharge and charge) at 11 kW
cycle_loss_43 = 0.000167  # Cycle loss per full cycle (100% DoD discharge and charge) at 43 kW

calendar_aging_0 = 0.0065  # Calendar aging per year if battery at 0% SoC
calendar_aging_40 = 0.0293  # Calendar aging per year if battery at 40% SoC
calendar_aging_90 = 0.065  # Calendar aging per year if battery at 90% SoC

sorted_soc_list = data
charging_power = 11
dt = 0.25
soh_emp = []
degradation = []

for j in range(1, 11):

    for i in range(1, len(data)):

        old_soc = sorted_soc_list[i - 1]
        new_soc = sorted_soc_list[i]

        # compute average for calendar aging
        avg_soc = (old_soc + new_soc) / 2

        # find the closest avg soc for calendar aging
        cal_soc = np.asarray([0, 40, 90])
        closest_index = np.abs(cal_soc - avg_soc).argmin()
        closest = cal_soc[closest_index]

        if closest == 0:
            cal_aging = calendar_aging_0 * dt / 8760
        elif closest == 40:
            cal_aging = calendar_aging_40 * dt / 8760
        elif closest == 90:
            cal_aging = calendar_aging_90 * dt / 8760
        else:
            cal_aging = None
            raise RuntimeError("Closest calendar aging SoC not recognised.")

        # calculate DoD of timestep
        dod = abs(new_soc - old_soc)

        # distinguish between high and low power charging according to input graph data
        if charging_power <= 22.0:
            cycle_loss = dod * cycle_loss_11 / 2  # convert to equivalent full cycles, that's why divided by 2
        else:
            cycle_loss = dod * cycle_loss_43 / 2  # convert to equivalent full cycles, that's why divided by 2

        # aggregate calendar and cyclic aging and append to degradation list
        degradation.append(cal_aging + cycle_loss)

        if (i % (8760 * 4 - 1) == 0):
            soh_emp.append(1 - sum(degradation))
        # print(f"emp soh: {self.soh}")

soh_emp = np.insert(soh_emp, 0, 1)
#plt.figure(figsize=[16,9])
a = (a[0], np.insert(a[1], 0, 1))
plt.plot(a[1])
plt.plot(soh_emp)
plt.grid(True)
plt.xlim([0,10])
plt.ylim([0.8, 1])
plt.legend(["SEI formation + Cycle counting", "Empirical linear degradation"])
plt.xticks(range(11))
plt.yticks([0.8,0.85,0.9,0.95,1.0])
plt.savefig("sei_emp_10.pdf")
plt.show()
