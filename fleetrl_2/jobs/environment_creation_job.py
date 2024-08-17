import logging
from pathlib import Path

from tidysci.package_manager import PackageManager
from tidysci.task import Task
from tidysci.task import register
from tidysci.util.types import external_path

from fleetrl_2.jobs.environment_dataset_job import EnvironmentDatasetJob
from fleetrl_2.jobs.reward_function_job import RewardFunctionJob
from fleetrl_2.jobs.site_parameters_job import SiteParametersJob

logger = logging.getLogger(PackageManager.get_name())


@register(alias=True)
class EnvironmentCreationJob(Task):

    def __init__(self,
                 _fleet_rl_dataset_: external_path,
                 _reward_function_job: external_path,
                 _site_parameters_job: external_path,
                 _dir_root: str,
                 rng_seed: int):
        super().__init__(_dir_root, rng_seed)

        self._fleet_rl_dataset_path = Path(_fleet_rl_dataset_)
        self._reward_function_path = Path(
            _reward_function_job)
        self._site_parameters_path = Path(
            _site_parameters_job)

        self.reward_function_job: RewardFunctionJob
        self.site_parameters_job: SiteParametersJob
        self.fleet_rl_dataset: EnvironmentDatasetJob

    def _setup(self):
        self.reward_function_job = RewardFunctionJob.from_task_directory(
            self._reward_function_path)
        self.site_parameters_job = SiteParametersJob.from_task_directory(
            self._site_parameters_path)
        self.fleet_rl_dataset = EnvironmentDatasetJob.from_task_directory(
            self._fleet_rl_dataset_path)

    def _run(self) -> None:
        # create env
        pass

    def is_finished(self) -> bool:
        return True


class _Episode:
    def __init__(self,
                 episode_length: int = 24,  # in hours
                 end_cutoff_days: int = 60,  # todo describe
                 price_lookahead: int = 8,  # in hours, maximum 12
                 building_load_lookahead: int = 4,  # in hours
                 pv_production_lookahead: int = 4,  # in hours
                 ):
        self.episode_length = episode_length
        self.end_cutoff_days = end_cutoff_days
        self.price_lookahead = price_lookahead
        self.building_load_lookahead = building_load_lookahead
        self.pv_production_lookahead = pv_production_lookahead
