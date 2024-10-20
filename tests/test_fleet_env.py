import pytest

from fleetrl.fleet_env.fleet_environment_v2 import FleetEnv, ModelTimeUnit
from fleetrl_2.jobs.environment_creation_job import MiscParams
from fleetrl_2.jobs.environment_dataset_job import EnvironmentDatasetJob
from fleetrl_2.jobs.reward_function_job import RewardFunctionJob


@pytest.fixture
def test_case_last_mile_delivery():

    env_dataset: EnvironmentDatasetJob = None
    reward_function: RewardFunctionJob = None
    log_data: bool = True
    normalization_strategy: str = None
    misc_params: MiscParams = None
    auxiliary_data: bool = True
    verbose: bool = False
    model_step_size: int = None
    model_time_unit: ModelTimeUnit = None

    input_list = [env_dataset,
                  reward_function,
                  log_data,
                  normalization_strategy,
                  misc_params,
                  auxiliary_data,
                  verbose,
                  log_data,
                  model_step_size,
                  model_time_unit]
    return input_list


def test_successful_env_creation(test_case_last_mile_delivery):
    env = FleetEnv(*test_case_last_mile_delivery)
    assert isinstance(env, FleetEnv)