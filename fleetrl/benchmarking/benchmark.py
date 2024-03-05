import pandas as pd


class Benchmark:
    """
    Parent class for benchmark modules.
    """

    def run_benchmark(self,
                      use_case: str,
                      env_kwargs: dict,
                      seed: int = None) -> pd.DataFrame:
        """
        This method contains the logic of the respective benchmarks, executes
        it on the given environment and returns a log.

        :param use_case: String that specifies use-case ("lmd", "ct", "ut")
        :param env_kwargs: Environment parameters
        :param seed: seed for RNG
        :return: Log Dataframe of the benchmark, can be saved as pickle
        """

        raise NotImplementedError("This is an abstract class.")

    def plot_benchmark(self,
                       log: pd.DataFrame,
                       ) -> None:
        """

        :param log: Log dataframe
        :return: None, plots the benchmark
        """
        raise NotImplementedError("This is an abstract class.")
