# forest_options.py
import sys
sys.path.append('..')

from typing import List
import threading

from tree import TreeOptions  # Assuming TreeOptions is defined in a separate file
from sampling import SamplingOptions  # Assuming SamplingOptions is defined in a separate file

class ForestOptions:
    def __init__(self, num_trees, ci_group_size, sample_fraction, mtry, min_node_size,
                 honesty, honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty,
                 num_threads, random_seed, sample_clusters, samples_per_cluster):
        self.ci_group_size = ci_group_size
        self.sample_fraction = sample_fraction
        self.tree_options = TreeOptions(mtry, min_node_size, honesty, honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty)
        self.sampling_options = SamplingOptions(samples_per_cluster, sample_clusters)
        self.random_seed = random_seed

        self.num_threads = self.validate_num_threads(num_threads)

        # If necessary, round the number of trees up to a multiple of the confidence interval group size.
        self.num_trees = num_trees + (num_trees % ci_group_size)

        if ci_group_size > 1 and sample_fraction > 0.5:
            raise RuntimeError("When confidence intervals are enabled, the sampling fraction must be less than 0.5.")

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
        if num_threads == threading.DEFAULT_THREAD_NUM:
            return threading.cpu_count()
        elif num_threads > 0:
            return num_threads
        else:
            raise RuntimeError("A negative number of threads was provided.")
