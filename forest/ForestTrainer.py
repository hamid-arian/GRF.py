import concurrent.futures

import random

import numpy as np

from forest.Forest import Forest
from sampling.RandomSampler import RandomSampler
from tree.TreeTrainer import TreeTrainer


class ForestTrainer:
    def __init__(self, relabeling_strategy, splitting_rule_factory, prediction_strategy):
        """
        Initialize a ForestTrainer.

        :param relabeling_strategy: Strategy for relabeling nodes.
        :param splitting_rule_factory: Factory for creating splitting rules.
        :param prediction_strategy: Strategy for making predictions.
        """
        self.tree_trainer = TreeTrainer(relabeling_strategy, splitting_rule_factory, prediction_strategy)

    def train(self, data, options):
        trees = self.train_trees(data, options)

        num_variables = data.get_num_cols() - len(data.get_disallowed_split_variables())
        ci_group_size = options.get_ci_group_size()
        return Forest(trees, num_variables, ci_group_size)

    def train_trees(self, data, options):
        num_samples = data.get_num_rows()
        num_trees = options.get_num_trees()

        # Validate sample and honesty fractions
        tree_options = options.get_tree_options()
        honesty = tree_options.get_honesty()
        honesty_fraction = tree_options.get_honesty_fraction()
        if num_samples * options.get_sample_fraction() < 1:
            raise RuntimeError("The sample fraction is too small, as no observations will be sampled.")
        elif honesty and (num_samples * options.get_sample_fraction() * honesty_fraction < 1 or
                          num_samples * options.get_sample_fraction() * (1 - honesty_fraction) < 1):
            raise RuntimeError("The honesty fraction is too close to 1 or 0, as no observations will be sampled.")

        # Calculate thread ranges for parallel execution
        num_groups = num_trees // options.get_ci_group_size()
        thread_ranges = self.split_sequence(0, num_groups - 1, options.get_num_threads())

        trees = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=options.get_num_threads()) as executor:
            futures = [executor.submit(self.train_batch, start_index, thread_ranges[i + 1] - start_index, data, options)
                       for i, start_index in enumerate(thread_ranges[:-1])]
            for future in concurrent.futures.as_completed(futures):
                trees.extend(future.result())

        return trees

    def split_sequence(self, start, end, num_parts):
        """
        Split a sequence into roughly equal parts.

        :param start: The starting index of the sequence.
        :param end: The ending index of the sequence.
        :param num_parts: The number of parts to split into.
        :return: A list of indices marking the start of each part.
        """
        part_length = (end - start + 1) // num_parts
        return [start + i * part_length for i in range(num_parts)] + [end + 1]

    def train_batch(self, start, num_trees, data, options):
        random.seed(options.get_random_seed() + start)
        trees = []
        ci_group_size = options.get_ci_group_size()

        for _ in range(num_trees):
            sampler = RandomSampler(random.randint(0, 2**32 - 1), options.get_sampling_options())
            if ci_group_size == 1:
                tree = self.tree_trainer.train(data, sampler,options.get_tree_options())
                trees.append(tree)
            else:
                group = self.train_ci_group(data, sampler, options)
                trees.extend(group)
        return trees

    def train_tree(self, data, sampler, options):
        """
        Train a single tree.

        :param data: The dataset used for training.
        :param sampler: The RandomSampler used for sampling data points.
        :param options: The ForestOptions providing configuration for training.
        :return: A trained Tree object.
        """
        clusters = sampler.sample_clusters(data.get_num_rows(), options.get_sample_fraction())
        return self.tree_trainer.train(data, sampler, clusters, options.get_tree_options())

    def train_ci_group(self, data, sampler, options):
        """
        Train a group of trees for confidence interval estimation.

        :param data: The dataset used for training.
        :param sampler: The RandomSampler used for sampling data points.
        :param options: The ForestOptions providing configuration for training.
        :return: A list of trained Tree objects.
        """
        trees = []

        num_rows = data.get_num_rows()
        if num_rows is None:
            raise ValueError("data.get_num_rows() returned None")

        indices = np.arange(num_rows).tolist()

        clusters = sampler.sample_clusters(num_rows, 0.5, indices)
        if clusters is None:
            raise ValueError("sampler.sample_clusters() returned None")

        # indices = np.arange(data.get_num_rows()).tolist()
        # clusters = sampler.sample_clusters(data.get_num_rows(), 0.5, indices)
        sample_fraction = options.get_sample_fraction()

        for _ in range(options.get_ci_group_size()):
            subsamples = np.random.choice(indices, size=int(len(indices) * sample_fraction * 2), replace=False).tolist()
            cluster_subsample = sampler.subsample(clusters, sample_fraction * 2, subsamples)
            tree = self.tree_trainer.train(data, sampler, cluster_subsample, options.get_tree_options())
            trees.append(tree)

        return trees


