import pandas as pd

class Evaluation:

    def evaluate_agent(self, env_kwargs: dict, norm_stats_path: str, model_path: str, seed: int = None):

        raise NotImplementedError("This is an abstract class.")

    def compare(self, rl_log, benchmark_log):

        raise NotImplementedError("This is an abstract class")

    def plot_soh(self, rl_log, benchmark_log):

        raise NotImplementedError("This is an abstract class")

    def plot_violations(self, rl_log, benchmark_log):

        raise NotImplementedError("This is an abstract class")

    def plot_action_dist(self, rl_log, benchmark_log):

        raise NotImplementedError("This is an abstract class")

    def plot_detailed_actions(self,
                              start_date: str | pd.Timestamp,
                              end_date: str | pd.Timestamp,
                              rl_log: pd.DataFrame=None,
                              uc_log: pd.DataFrame=None,
                              dist_log: pd.DataFrame=None,
                              night_log: pd.DataFrame=None):

        raise NotImplementedError("This is an abstract class")

    @staticmethod
    def _change_param(env_kwargs: dict, key: str, val) -> dict:

        raise NotImplementedError("This is an abstract class")

    def _get_from_obs(self, log: dict) -> pd.DataFrame:

        raise NotImplementedError("This is an abstract class")