from fleetrl.fleet_env.episode import Episode

class EventManager:
    """
    The EventManager class decides if a new event occurred that is relevant for the RL agent
    For example: Subsequent steps where no new car arrives are flagged as irrelevant and not sent to the agent
    New arrival triggers an observation to be sent to the RL agent via the step() return function

    Idea from: https://doi.org/10.1016/j.buildenv.2023.110546
    """

    def __init__(self):
        pass

    @staticmethod
    def check_event(episode: Episode) -> bool:
        """
        :param events: variable that contains number of relevant events
        :return: bool to confirm that a relevant event took place
        """

        if (episode.time.minute == 15) and (episode.time.second == 0):
            episode.events += 1

        # observation
        # check_event(skip(action))
        # check_event gets the NEXT obs that would happen if step is skipped

        # check if a new car arrived

        return episode.events > 0
