Note: I am currently finishing my thesis. Once it is done, I will make a tutorial, document everything properly, as well as clean up any unnecessary files.
For now, try the complete_pipeline notebook - it can generate new schedules and train agents with a few lines of code.

Please view the license below.

This framework provides a realistic Reinforcement Learning
environment for EV charging and is focused on commercial vehicle
fleets.

It includes the following features:
- Variable time resolution
- Variable episode length, overnight charging
- ENTSO-E price data
- PV data from MERRA-2 dataset
- Schedule generator
- 3 different use-cases of EV fleets included

<img width="600" src="https://github.com/EnzoCording/FleetRL/blob/master/FleetRL_overview.jpg">

Right now the following is working:
- Step function charges EVs
- Calculates cost based on spot market
- Checks for SOC and grid violations
- Returns next soc and time left at the charger

data_processing:
- Loads schedules from emobpy output
- Computes important metrics such as soc on arrival and time_left
- Use of pd.merge_asof() to significantly speed up the data processing

fleet_environment:
- The implementation of the gym Env class for FleetRL
- Ties in together all functionalities of the environment in the init, reset, and step function

load_calculation:
- Calculates violations of building load
- Builds pandapower grid and calculates transformer loading, as well as phase angle

Citation:
Please reference this project if you find it useful for your research.

[License](LICENSE)

No custom license is added to this repository until the thesis has been published.
Until then, the copyright remains solely with the author - please contact for any further questions.
Copyright (c) 2023, Enzo Alexander Cording - https://github.com/EnzoCording/FleetRL
