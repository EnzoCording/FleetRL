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
