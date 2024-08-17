from abc import ABC
from abc import abstractmethod

import pandas as pd

from fleetrl_2.jobs.schedule_parameters.schedule_parameters import Charger
from fleetrl_2.jobs.schedule_parameters.schedule_parameters import Consumption
from fleetrl_2.jobs.schedule_parameters.schedule_parameters import DepartureTime
from fleetrl_2.jobs.schedule_parameters.schedule_parameters import DistanceTravelled
from fleetrl_2.jobs.schedule_parameters.schedule_parameters import ReturnTime


class ScheduleGenerator(ABC):
    """
    Probabilistic schedule generation for EVs.
    The format should be kept in-line to emobpy to
    enable compatability and ease of use. Refer to docs for more info.
    """

    def __init__(self,
                 starting_date: str,
                 ending_date: str,
                 freq: str,
                 num_evs: int,
                 consumption: Consumption,
                 charger: Charger,
                 return_time: ReturnTime,
                 departure_time: DepartureTime,
                 distance_travelled: DistanceTravelled,
                 seed: int,
                 **kwargs):
        self.starting_date = starting_date
        self.ending_date = ending_date
        self.freq = freq
        self.num_evs = num_evs
        self.consumption = consumption
        self.charger = charger
        self.return_time = return_time
        self.departure_time = departure_time
        self.distance_travelled = distance_travelled
        self.seed = seed
        for k, v in kwargs.items():
            setattr(self, k, v)

    def generate(self) -> pd.DataFrame:
        """
        TODO MULTICORE
        - Easiest distribution: one ev per core
        - downside: 1 EV schedules are generated on one core and its slow
        - if i split it per month or number of rows, i need to ensure coherence
        - so that the first entry of core 2 matches the last entry of core 1
        - splitting cores inbetween trips is not a good idea
        - times where the car is there or not depends on the schedule algo
        - maybe there is a way to split it and recognising when a trip is done

        """
        schedules = []
        for i in range(self.num_evs):
            schedules.append(self._generate(i))

        return pd.concat(schedules, axis=0)

    @abstractmethod
    def _generate(self, ev_id: int) -> pd.DataFrame:

        """
        Generate a schedule.

        :param ev_id: Identifier of vehicle.
        :return: pd.DataFrame of the schedule.
        """
