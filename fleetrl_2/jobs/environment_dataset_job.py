import logging
from pathlib import Path

from tidysci.package_manager import PackageManager
from tidysci.task import Task
from tidysci.task import register
from tidysci.util.types import external_path

from fleetrl_2.jobs.ev_config_job import EvConfigJob
from fleetrl_2.jobs.schedule_generation_job import ScheduleGenerationJob
from fleetrl_2.jobs.site_parameters_job import SiteParametersJob

logger = logging.getLogger(PackageManager.get_name())


@register(alias=True)
class EnvironmentDatasetJob(Task):

    def __init__(self,
                 _schedule_generation_job: external_path,
                 _site_parameters_job: external_path,
                 _ev_config_job: external_path,
                 config_jobs: dict,
                 auxiliary_data: dict,
                 _dir_root: str,
                 rng_seed: int):

        super().__init__(_dir_root, rng_seed)

        self.config_jobs = _ConfigJobs(**config_jobs)
        self.auxiliary_data = _AuxiliaryData(**auxiliary_data)

        self._schedule_generation_path = Path(_schedule_generation_job)
        self._site_parameters_path = Path(_site_parameters_job)
        self._ev_config_path = Path(_ev_config_job)

        self.site_parameters_job: SiteParametersJob
        self.schedule_generation_job: ScheduleGenerationJob
        self.ev_config_job: EvConfigJob

    def _setup(self) -> None:
        super()._setup()

        self.site_parameters_job = (
            SiteParametersJob.from_task_directory(
                self._site_parameters_path))

        self.schedule_generation_job = (
            ScheduleGenerationJob.from_task_directory(
                self._schedule_generation_path))

        self.ev_config_job = (
            EvConfigJob.from_task_directory(
                self._ev_config_path))

    def is_finished(self) -> bool:
        return True


class _AuxiliaryData:
    def __init__(self,
                 _external_building_load: str | None = None,
                 _external_pv_production: str | None = None):
        self._external_building_load = Path(
            _external_building_load) if _external_building_load is not None else None
        self._external_pv_production = Path(
            _external_pv_production) if _external_pv_production is not None else None


class _ConfigJobs:
    def __init__(self,
                 _external_ev_config_job="path"
                 ):
        self._external_ev_config_job = Path(_external_ev_config_job)
