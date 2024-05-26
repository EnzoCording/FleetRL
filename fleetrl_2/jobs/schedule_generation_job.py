import importlib
import logging
from pathlib import Path
from typing import Type

import pandas as pd
from pyjob_todo.job import Job
from pyjob_todo.job import registered_job
from pyjob_todo.package_manager import PackageManager

from fleetrl.utils.schedule_generation.schedule_generator import \
    ScheduleGenerator
from schedule_statistics_job import ScheduleStatisticsJob

logger = logging.getLogger(PackageManager.get_name())


@registered_job
class ScheduleGenerationJob(Job):
    def __init__(self,
                 schedule_frequency: str,
                 num_evs: int,
                 _external_schedule_csv: str,
                 _external_schedule_job: str,
                 schedule_output_name: str,
                 starting_date: str,
                 ending_date: str,
                 schedule_algorithm: str,
                 schedule_generation_kwargs: dict,
                 **kwargs):

        super().__init__(**kwargs)

        self.schedule_frequency = schedule_frequency
        self.num_evs = num_evs

        # TODO make pd Date Object and check it works
        self.starting_date = starting_date
        self.ending_date = ending_date

        self.schedule_algorithm = schedule_algorithm
        self.schedule_generation_kwargs = schedule_generation_kwargs

        if ((not _external_schedule_job and not _external_schedule_csv)
                or (_external_schedule_job and _external_schedule_csv)):
            raise ValueError(
                "Must specify either a schedule job or a schedule csv file "
                "- not neither or both.")

        self._external_schedule_csv = (Path(_external_schedule_csv)
                                       if _external_schedule_csv else None)
        if (self._external_schedule_csv is not None
                and (not self._external_schedule_csv.is_file())):
            raise ValueError(
                "Invalid path to external schedule csv file: "
                f"{_external_schedule_csv}")

        self._external_schedule_job = (Path(_external_schedule_job)
                                       if _external_schedule_job else None)

        if not schedule_output_name.endswith(".csv"):
            schedule_output_name += ".csv"

        self.schedule_output_name = schedule_output_name

        self.schedule_output_csv: Path
        self.schedule_statistics_job: ScheduleStatisticsJob
        self.schedule_generator: ScheduleGenerator

    @staticmethod
    def get_toml_key() -> str:
        return "schedule_generation"

    def _setup(self) -> None:
        if self._external_schedule_csv is not None:
            self.schedule_output_csv = self._external_schedule_csv
        else:
            self.schedule_statistics_job: ScheduleStatisticsJob = Job.from_job_directory(
                self._external_schedule_job)

            try:
                # extract module and class of algorithm implementation
                module_name, class_name = self.schedule_algorithm.split('.')
                # load module
                module = importlib.import_module(
                    f"fleetrl.utils.schedule_generation.{module_name}")
                # get the specific class of the schedule generator
                _class: Type[ScheduleGenerator] = getattr(module, class_name)
                # instantiate algorithm object with all needed paramaters
                self.schedule_generator = _class(
                    starting_date=self.starting_date,
                    ending_date=self.ending_date,
                    freq=self.schedule_frequency,
                    num_evs=self.num_evs,
                    consumption=self.schedule_statistics_job.consumption,
                    charger=self.schedule_statistics_job.charger,
                    return_time=self.schedule_statistics_job.return_time,
                    departure_time=self.schedule_statistics_job.departure_time,
                    distance_travelled=self.schedule_statistics_job.distance_travelled,
                    seed=self.seed,
                    **self.schedule_generation_kwargs)
            except ImportError as e:
                raise ValueError(
                    f"Invalid schedule algorithm: {self.schedule_algorithm}.\n"
                    "Did you create the class "
                    "fleetrl.utils.schedule_generation."
                    f"{self.schedule_algorithm}?"
                ) from e

    def is_finished(self) -> bool:
        return (self.schedule_output_csv is not None
                and self.schedule_output_csv.is_file())

    def _run(self):
        self._generate_schedule()

    def _exit(self) -> None:
        # check if csv is readable by pandas
        pd.read_csv(self.schedule_output_csv, nrows=20)

    def _generate_schedule(self):
        logger.info("Generating schedules... This may take a while.")
        complete_schedule = self.schedule_generator.generate()

        # os independent path with pathlib
        p = self._dir_root / self.schedule_output_name
        complete_schedule.to_csv(p)
        logger.info(f"Schedule generation complete. Saved in: {p}")
        self.schedule_name = p
