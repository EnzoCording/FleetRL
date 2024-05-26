import pandas as pd

from fleetrl.utils.schedule_generation.schedule_generator import \
    ScheduleGenerator


class DeliveryScheduleGenerator(ScheduleGenerator):
    def _generate(self, ev_id: int) -> pd.DataFrame:
        pass
