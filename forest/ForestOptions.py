import multiprocessing

from sampling.SamplingOptions import SamplingOptions
from tree.TreeOptions import TreeOptions


class ForestOptions:
    DEFAULT_NUM_THREADS = 0

    def __init__(self, num_trees, ci_group_size, sample_fraction, mtry, min_node_size, honesty, honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty, num_threads, random_seed, sample_clusters, samples_per_cluster):
        """
        Initialize ForestOptions.

        :param num_trees: The number of trees in the forest.
        :param ci_group_size: The group size for confidence interval calculation.
        :param sample_fraction: The fraction of samples to draw for training each tree.
        :param mtry: The number of variables tried at each split.
        :param min_node_size: The minimum size of a node.
        :param honesty: Whether to use honesty when building trees.
        :param honesty_fraction: The fraction of data to be used for honesty in tree building.
        :param honesty_prune_leaves: Whether to prune leaves based on honesty.
        :param alpha: The alpha parameter for tree building.
        :param imbalance_penalty: The imbalance penalty for tree building.
        :param num_threads: The number of threads for parallel execution.
        :param random_seed: The random seed for the random number generator.
        :param sample_clusters: The clusters of samples.
        :param samples_per_cluster: The number of samples per cluster.
        """
        self.ci_group_size = ci_group_size
        self.sample_fraction = sample_fraction
        self.tree_options = TreeOptions(mtry, min_node_size, honesty, honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty)
        self.sampling_options = SamplingOptions(samples_per_cluster, sample_clusters)
        self.random_seed = random_seed

        self.num_threads = self.validate_num_threads(num_threads)

        # Round the number of trees up to a multiple of the CI group size
        self.num_trees = num_trees + (num_trees % ci_group_size)

        if ci_group_size > 1 and sample_fraction > 0.5:
            raise ValueError("When confidence intervals are enabled, the sampling fraction must be less than 0.5.")

    def get_num_trees(self):
        return self.num_trees

    def get_ci_group_size(self):
        return self.ci_group_size

    def get_sample_fraction(self):
        return self.sample_fraction

    def get_tree_options(self):
        return self.tree_options

    def get_sampling_options(self):
        return self.sampling_options

    def get_num_threads(self):
        return self.num_threads

    def get_random_seed(self):
        return self.random_seed

    @staticmethod
    def validate_num_threads(num_threads):
        if num_threads == ForestOptions.DEFAULT_NUM_THREADS:
            return multiprocessing.cpu_count()
        elif num_threads > 0:
            return num_threads
        else:
            raise ValueError("A negative number of threads was provided.")

