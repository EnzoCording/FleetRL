import pandas as pd

from FleetRL.fleet_env.fleet_environment import FleetEnv
from FleetRL.utils.schedule_generator.schedule_generator import ScheduleGenerator, ScheduleType

# initiate FleetEnv instance to access its modules
env = FleetEnv()
schedule_generator_lmd: ScheduleGenerator = ScheduleGenerator("./", schedule_type=ScheduleType.Delivery, file_comment="one_ev_lmd")
schedule_generator_ct: ScheduleGenerator = ScheduleGenerator("./", schedule_type=ScheduleType.Caretaker, file_comment="one_ev_ct")
schedule_generator_ut: ScheduleGenerator = ScheduleGenerator("./", schedule_type=ScheduleType.Utility, file_comment="one_ev_ut")

lmd = schedule_generator_lmd.generate_schedule()
ct = schedule_generator_ct.generate_schedule()
ut = schedule_generator_ut.generate_schedule()

schedule_generator_lmd: ScheduleGenerator = ScheduleGenerator("./", schedule_type=ScheduleType.Delivery, file_comment="one_ev_lmd", save_schedule=False)
schedule_generator_ct: ScheduleGenerator = ScheduleGenerator("./", schedule_type=ScheduleType.Caretaker, file_comment="one_ev_ct", save_schedule=False)
schedule_generator_ut: ScheduleGenerator = ScheduleGenerator("./", schedule_type=ScheduleType.Utility, file_comment="one_ev_ut", save_schedule=False)

lmd = []
ct = []
ut = []

for i in range(50):
    lmd.append(schedule_generator_lmd.generate_schedule())
    ct.append(schedule_generator_ct.generate_schedule())
    ut.append(schedule_generator_ut.generate_schedule())

lmd_df = pd.concat(lmd)
ct_df = pd.concat(ct)
ut_df = pd.concat(ut)

lmd_df.to_csv("50_lmd.csv")
ct_df.to_csv("50_ct.csv")
ut_df.to_csv("50_ut.csv")