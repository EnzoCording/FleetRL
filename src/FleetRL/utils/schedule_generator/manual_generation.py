from FleetRL.fleet_env.fleet_environment import FleetEnv
from FleetRL.utils.schedule_generator.schedule_generator import ScheduleGenerator, ScheduleType, ScheduleConfig

env = FleetEnv()

schedule_generator: ScheduleGenerator = ScheduleGenerator("./", schedule_type=ScheduleType.Utility, file_comment="one_ev_ut")
sched = schedule_generator.generate_schedule()
