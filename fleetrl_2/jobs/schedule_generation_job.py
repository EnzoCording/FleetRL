import importlib
import logging
from pathlib import Path
from typing import Type

import pandas as pd
from tomlchef.job import Job
from tomlchef.job import registered_job
from tomlchef.package_manager import PackageManager

from fleetrl.utils.schedule_generation.schedule_generator import \
    ScheduleGenerator
from .schedule_parameters_job import ScheduleParametersJob

logger = logging.getLogger(PackageManager.get_name())


@registered_job
class ScheduleGenerationJob(Job):
    def __init__(self,
                 schedule_frequency: str,
                 num_evs: int,
                 _external_schedule_csv: str,
                 _external_schedule_parameters_job: str,
                 schedule_output_name: str,
                 starting_date: str,
                 ending_date: str,
                 **kwargs):

        super().__init__(**kwargs)

        self.schedule_frequency = schedule_frequency
        self.num_evs = num_evs

        # TODO make pd Date Object and check it works
        self.starting_date = starting_date
        self.ending_date = ending_date

        if ((not _external_schedule_parameters_job
             and not _external_schedule_csv)
                or (_external_schedule_parameters_job
                    and _external_schedule_csv)):
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

        self._external_schedule_parameters_job = (
            Path(_external_schedule_parameters_job)
            if _external_schedule_parameters_job else None)

        if not schedule_output_name.endswith(".csv"):
            schedule_output_name += ".csv"

        self.schedule_output_name = schedule_output_name

        self.schedule_algorithm: str
        self.schedule_output_csv: Path | None = None
        self.schedule_parameters_job: ScheduleParametersJob
        self.schedule_generator: ScheduleGenerator

    @staticmethod
    def get_toml_key() -> str:
        return "schedule_generation"

    def _setup(self) -> None:
        if self._external_schedule_csv is not None:
            self.schedule_output_csv = self._external_schedule_csv
        else:
            self.schedule_parameters_job = (
                Job.from_job_directory(
                    self._external_schedule_parameters_job)
            )
            self.schedule_algorithm = self.schedule_parameters_job.schedule_algorithm

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
                    consumption=self.schedule_parameters_job.consumption,
                    charger=self.schedule_parameters_job.charger,
                    return_time=self.schedule_parameters_job.return_time,
                    departure_time=self.schedule_parameters_job.departure_time,
                    distance_travelled=self.schedule_parameters_job.distance_travelled,
                    seed=self.seed)
                    #**self.schedule_generation_kwargs)
                pass
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
        self.schedule_output_csv = p
