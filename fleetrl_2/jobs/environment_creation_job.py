import logging
from pathlib import Path

from tidysci.package_manager import PackageManager
from tidysci.task import Task
from tidysci.task import register

from fleetrl_2.jobs.environment_dataset_job import EnvironmentDatasetJob
from fleetrl_2.jobs.reward_function_job import RewardFunctionJob

logger = logging.getLogger(PackageManager.get_name())


@register(alias=True)
class EnvironmentCreationJob(Task):

    def __init__(self,
                 _environment_dataset_job: str,
                 _reward_function_job: str,
                 use_auxiliary_data: bool,
                 normalization_strategy: str,
                 episode: dict,
                 misc: dict,
                 _dir_root: str,
                 rng_seed: int):
        super().__init__(_dir_root, rng_seed)

        self._environment_dataset_path = Path(_environment_dataset_job)
        self._environment_dataset_job: EnvironmentDatasetJob
        self._reward_function_path = Path(_reward_function_job)
        self._reward_function_job: RewardFunctionJob

        self.use_auxiliary_data = use_auxiliary_data
        self.normalization_strategy = normalization_strategy
        self.episode = _Episode(**episode)
        self.misc = _Misc(**misc)

    def _setup(self):
        self._environment_dataset_job = (
            EnvironmentDatasetJob.from_task_directory(
                self._environment_dataset_path))

        self._reward_function_job = RewardFunctionJob.from_task_directory(
            self._reward_function_path
        )

    def _run(self) -> None:
        # todo create env
        pass

    def is_finished(self) -> bool:
        return True  # todo


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


class _Misc:
    def __init__(self, battery_degradation_type: str):
        self.battery_degradation_type = battery_degradation_type
