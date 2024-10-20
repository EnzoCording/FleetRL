from parso.normalizer import Normalizer

from fleetrl.utils.normalization.normalization import Normalization
from fleetrl.utils.observation.observer import Observer
from fleetrl.utils.observation.observer_bl_pv import ObserverWithBoth
from fleetrl.utils.observation.observer_price_only import ObserverPriceOnly
from fleetrl.utils.observation.observer_soc_time_only import ObserverSocTimeOnly
from fleetrl.utils.observation.observer_with_building_load import ObserverWithBuildingLoad
from fleetrl.utils.observation.observer_with_pv import ObserverWithPV
from fleetrl.utils.time_picker.eval_time_picker import EvalTimePicker
from fleetrl.utils.time_picker.random_time_picker import RandomTimePicker
from fleetrl.utils.time_picker.static_time_picker import StaticTimePicker
from fleetrl.utils.time_picker.time_picker import TimePicker
from fleetrl_2.jobs.environment_creation_job import EpisodeParams


def detect_dim_and_bounds(include_price: bool,
                          num_cars: int,
                          aux_flag: bool,
                          normalizer: Normalization,
                          include_pv: bool,
                          include_building_load: bool,
                          price_lookahead: int,
                          building_load_lookahead: int):
    """
    This function chooses the right dimension of the observation space based on the chosen configuration.
    Each increase of dim is explained below. The low_obs and high_obs are built in the normalizer object,
    using the dim value that was calculated in this function.

    - set boundaries of the observation space, detects if normalized or not.
    - If aux flag is true, additional information enlarges the observation space.
    - The following code goes through all possible environment setups.
    - Depending on the setup, the dimensions differ and every case is handled differently.

    :return: low_obs and high_obs: tuple[float, float] | tuple[np.ndarray, np.ndarray] -> used for gym.Spaces
    """

    if not include_price:
        dim = 2 * num_cars  # soc and time left for each EV
        if aux_flag:
            dim += num_cars  # there
            dim += num_cars  # target soc
            dim += num_cars  # charging left
            dim += num_cars  # hours needed
            dim += num_cars  # laxity
            dim += 1  # evse power
            dim += 6  # month, week, hour sin/cos
        low_obs, high_obs = normalizer.make_boundaries((dim,))

    elif not include_building_load and not include_pv:
        dim = 2 * num_cars + (price_lookahead + 1) * 2
        if aux_flag:
            dim += num_cars  # there
            dim += num_cars  # target soc
            dim += num_cars  # charging left
            dim += num_cars  # hours needed
            dim += num_cars  # laxity
            dim += 1  # evse power
            dim += 6  # month, week, hour sin/cos
        low_obs, high_obs = normalizer.make_boundaries((dim,))

    elif include_building_load and not include_pv:
        dim = (2 * num_cars
               + (price_lookahead + 1) * 2
               + building_load_lookahead + 1
               )
        if aux_flag:
            dim += num_cars  # there
            dim += num_cars  # target soc
            dim += num_cars  # charging left
            dim += num_cars  # hours needed
            dim += num_cars  # laxity
            dim += 1  # evse power
            dim += 1  # grid cap
            dim += 1  # avail grid cap for charging
            dim += 1  # possible avg action per car
            dim += 6  # month, week, hour sin/co
        low_obs, high_obs = normalizer.make_boundaries((dim,))

    elif not include_building_load and include_pv:
        dim = (2 * num_cars
               + (price_lookahead + 1) * 2
               + building_load_lookahead + 1
               )
        if aux_flag:
            dim += num_cars  # there
            dim += num_cars  # target soc
            dim += num_cars  # charging left
            dim += num_cars  # hours needed
            dim += num_cars  # laxity
            dim += 1  # evse power
            dim += 6  # month, week, hour sin/cos
        low_obs, high_obs = normalizer.make_boundaries((dim,))

    elif include_building_load and include_pv:
        dim = (2 * num_cars  # soc and time left
               + (price_lookahead + 1) * 2  # price and tariff
               + 2 * (building_load_lookahead + 1)  # pv and building load
               )
        if aux_flag:
            dim += num_cars  # there
            dim += num_cars  # target soc
            dim += num_cars  # charging left
            dim += num_cars  # hours needed
            dim += num_cars  # laxity
            dim += 1  # evse power
            dim += 1  # grid cap
            dim += 1  # avail grid cap for charging
            dim += 1  # possible avg action per car
            dim += 6  # month, week, hour sin/cos
        low_obs, high_obs = normalizer.make_boundaries((dim,))

    else:
        low_obs = None
        high_obs = None
        raise ValueError("Problem with environment setup. Check building and pv flags.")

    return low_obs, high_obs


def choose_observer(include_price: bool,
                        include_building_load: bool,
                        include_pv: bool):
    """
    This function chooses the right observer, depending on whether to include price, building, PV, etc.
    :return: obs (Observer) -> The observer module to choose
    """

    # All observations are made in the observer class
    # not even price: only soc and time left
    if not include_price:
        obs: Observer = ObserverSocTimeOnly()
    # only price
    elif not include_building_load and not include_pv:
        obs: Observer = ObserverPriceOnly()
    # price and building load
    elif include_building_load and not include_pv:
        obs: Observer = ObserverWithBuildingLoad()
    # price and pv
    elif not include_building_load and include_pv:
        obs: Observer = ObserverWithPV()
    # price, building load and pv
    elif include_building_load and include_pv:
        obs: Observer = ObserverWithBoth()
    else:
        raise TypeError("Observer configuration not found. Recheck flags.")

    return obs


def choose_time_picker(time_picker: str,
                       episode_length: int
                       ):

    """
    Chooses the right time picker based on the specified in input string.
    Static: Always the same time is picked to start an episode
    Random: Start an episode randomly from the training set
    Eval: Start an episode randomly from the validation set
    :param episode_length: Length of episode in hours
    :param time_picker: (string), specifies which time picker to choose: "static", "eval", "random"
    :return: tp (TimePicker) -> time picker object
    """

    # Load time picker module
    if time_picker == "STATIC":
        # when an episode starts, this class picks the same starting time
        tp: TimePicker = StaticTimePicker()
    elif time_picker == "EVAL":
        # picks a random starting times from test set (nov - dez)
        tp: TimePicker = EvalTimePicker(episode_length)
    elif time_picker == "RANDOM":
        # picks random starting times from training set (jan - oct)
        tp: TimePicker = RandomTimePicker()
    else:
        # must choose between static, eval or random
        raise TypeError("Time picker type not recognised")

    return tp
