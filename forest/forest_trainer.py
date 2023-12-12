# forest_trainer.py

import random
from concurrent.futures import ThreadPoolExecutor

from prediction.opt_prediction_strategy import OptimizedPredictionStrategy
from relabeling.relabeling_strategy import RelabelingStrategy
from splitting.factory.splitting_rule_factory import SplittingRuleFactory
from tree.tree_trainer import TreeTrainer
from tree.tree import Tree
from forest.forest import Forest
from forest.forest_options import ForestOptions
from commons.utility import split_sequence
from random_sampler import RandomSampler
from data import Data

class ForestTrainer:
    def __init__(self, relabeling_strategy, splitting_rule_factory, prediction_strategy):
        self.tree_trainer = TreeTrainer(relabeling_strategy, splitting_rule_factory, prediction_strategy)

    def train(self, data, options):
        trees = self.train_trees(data, options)

        num_variables = data.get_num_cols() - len(data.get_disallowed_split_variables())
        ci_group_size = options.get_ci_group_size()
        return Forest(trees, num_variables, ci_group_size)

    def train_trees(self, data, options):
        num_samples = data.get_num_rows()
        num_trees = options.get_num_trees()

        tree_options = options.get_tree_options()
        honesty = tree_options.get_honesty()
        honesty_fraction = tree_options.get_honesty_fraction()

        if num_samples * options.get_sample_fraction() < 1:
            raise ValueError("The sample fraction is too small, as no observations will be sampled.")
        elif honesty and (
            num_samples * options.get_sample_fraction() * honesty_fraction < 1
            or num_samples * options.get_sample_fraction() * (1 - honesty_fraction) < 1
        ):
            raise ValueError(
                "The honesty fraction is too close to 1 or 0, as no observations will be sampled."
            )

        num_groups = num_trees // options.get_ci_group_size()
        thread_ranges = []
        split_sequence(thread_ranges, 0, num_groups - 1, options.get_num_threads())

        trees = []
        with ThreadPoolExecutor(max_workers=options.get_num_threads()) as executor:
            futures = [
                executor.submit(self.train_batch, start, num_groups, data, options)
                for start in thread_ranges[:-1]
            ]

            for future in futures:
                thread_trees = future.result()
                trees.extend(thread_trees)

        return trees

    def train_batch(self, start, num_trees, data, options):
        ci_group_size = options.get_ci_group_size()
        random_number_generator = random.Random(options.get_random_seed() + start)
        trees = []

        for _ in range(num_trees):
            tree_seed = random_number_generator.randint(0, 2 ** 32 - 1)
            sampler = RandomSampler(tree_seed, options.get_sampling_options())

            if ci_group_size == 1:
                tree = self.train_tree(data, sampler, options)
                trees.append(tree)
            else:
                group = self.train_ci_group(data, sampler, options)
                trees.extend(group)

        return trees

    def train_tree(self, data, sampler, options):
        clusters = sampler.sample_clusters(data.get_num_rows(), options.get_sample_fraction())
        return self.tree_trainer.train(data, sampler, clusters, options.get_tree_options())

    def train_ci_group(self, data, sampler, options):
        trees = []

        clusters = sampler.sample_clusters(data.get_num_rows(), 0.5)
        sample_fraction = options.get_sample_fraction()

        for _ in range(options.get_ci_group_size()):
            cluster_subsample = sampler.subsample(clusters, sample_fraction * 2)
            tree = self.tree_trainer.train(data, sampler, cluster_subsample, options.get_tree_options())
            trees.append(tree)

        return trees
