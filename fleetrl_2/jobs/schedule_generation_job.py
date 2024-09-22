import importlib
import logging
from pathlib import Path
from typing import Type

import pandas as pd
from tidysci.package_manager import PackageManager
from tidysci.task import Task
from tidysci.task import register
from tidysci.util.types import external_path

from fleetrl.utils.schedule_generation.schedule_generator import (
    ScheduleGenerator)
from fleetrl_2.jobs.schedule_parameters.schedule_parameters import (
    ScheduleParameters)

logger = logging.getLogger(PackageManager.get_name())


@register(alias=True)
class ScheduleGenerationJob(Task):
    def __init__(self,
                 schedule_frequency: str,
                 num_evs: int,
                 _schedule_csv: str,
                 _schedule_parameters_path: str,
                 schedule_output_name: str,
                 starting_date: str,
                 ending_date: str,
                 _dir_root: str,
                 rng_seed: int):

        super().__init__(_dir_root, rng_seed)

        self.schedule_frequency = schedule_frequency
        self.num_evs = num_evs

        # TODO make pd Date Object and check it works
        self.starting_date = starting_date
        self.ending_date = ending_date

        if ((not _schedule_parameters_path
             and not _schedule_csv)
                or (_schedule_parameters_path
                    and _schedule_csv)):
            raise ValueError(
                "Must specify either a schedule job or a schedule csv file "
                "- not neither or both.")

        self._schedule_csv = (Path(_schedule_csv) if _schedule_csv else None)
        if (self._schedule_csv is not None
                and (not self._schedule_csv.is_file())):
            raise ValueError(
                f"Invalid path to external schedule csv file: {_schedule_csv}")

        self._schedule_parameters_path = (
            Path(_schedule_parameters_path)
            if _schedule_parameters_path else None)

        if not schedule_output_name.endswith(".csv"):
            schedule_output_name += ".csv"

        self.schedule_output_name = schedule_output_name

        self.schedule_algorithm: str
        self.schedule_output_csv: Path = self._dir_root / self.schedule_output_name
        self.schedule_parameters: ScheduleParameters
        self.schedule_generator: ScheduleGenerator

    def _setup(self) -> None:
        if self._schedule_csv is not None:
            self.schedule_output_csv = self._schedule_csv
        else:
            self.schedule_parameters = (
                ScheduleParameters.from_task_directory(
                    self._schedule_parameters_path)
            )
            self.schedule_algorithm = (
                self.schedule_parameters.schedule_algorithm)

            try:
                # extract module and class of algorithm implementation
                module_name, class_name = self.schedule_algorithm.split('.')
                # load module
                module = importlib.import_module(
                    f"fleetrl.utils.schedule_generation.{module_name}")
                # get the specific class of the schedule generator
                _class: Type[ScheduleGenerator] = getattr(module, class_name)
                # instantiate algorithm object with all needed parameters
                self.schedule_generator = _class(
                    starting_date=self.starting_date,
                    ending_date=self.ending_date,
                    freq=self.schedule_frequency,
                    num_evs=self.num_evs,
                    consumption=self.schedule_parameters.consumption,
                    charger=self.schedule_parameters.charger,
                    return_time=self.schedule_parameters.return_time,
                    departure_time=self.schedule_parameters.departure_time,
                    distance_travelled=self.schedule_parameters.distance_travelled,
                    seed=self.seed)
            except ImportError as e:
                raise ValueError(
                    f"Invalid schedule algorithm: {self.schedule_algorithm}.\n"
                    "Did you create the class "
                    "fleetrl.utils.schedule_generation."
                    f"{self.schedule_algorithm}?"
                ) from e

    def is_finished(self) -> bool:
        return self.schedule_output_csv.is_file()

    def _run(self):
        self._generate_schedule()

    def _exit(self) -> None:
        # check if csv is readable by pandas
        pd.read_csv(self.schedule_output_csv, nrows=20)

    def _generate_schedule(self):
        logger.info("Generating schedules... This may take a while.")
        complete_schedule = self.schedule_generator.generate()

        # os independent path with pathlib
        complete_schedule.to_csv(self.schedule_output_csv)
        logger.info("Schedule generation complete. Saved in: "
                    f"{self.schedule_output_csv}")
