import numpy as np
from collections import defaultdict

class SampleWeightComputer:
    def compute_weights(self, sample, forest, leaf_nodes_by_tree, valid_trees_by_sample):
        weights_by_sample = defaultdict(float)

        # Create a list of weighted neighbors for this sample.
        for tree_index in range(len(forest.get_trees())):
            if not valid_trees_by_sample[sample][tree_index]:
                continue

            leaf_nodes = leaf_nodes_by_tree[tree_index]
            node = leaf_nodes[sample]

            tree = forest.get_trees()[tree_index]
            samples = tree.get_leaf_samples()[node]

            if samples:
                self.add_sample_weights(samples, weights_by_sample)

        self.normalize_sample_weights(weights_by_sample)
        return weights_by_sample

    def add_sample_weights(self, samples, weights_by_sample):
        sample_weight = 1.0 / len(samples)

        for sample in samples:
            weights_by_sample[sample] += sample_weight

    def normalize_sample_weights(self, weights_by_sample):
        total_weight = sum(weights_by_sample.values())

        for sample in weights_by_sample:
            weights_by_sample[sample] /= total_weight
