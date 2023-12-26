

class EventManager:
    """
    The EventManager class decides if a new event occured that is relevant for the RL agent
    For example: Subsequent steps where no new car arrives are flagged as irrelevant and not sent to the agent
    New arrival triggers an observation to be sent to the RL agent via the step() return function

    Idea from: https://doi.org/10.1016/j.buildenv.2023.110546
    """

    def __init__(self):
        pass

    def check_event(self, observation):
        # observation
        # check_event(skip(action))
        # check_event gets the NEXT obs that would happen if step is skipped
        pass