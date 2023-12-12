import concurrent.futures
from collections import defaultdict
import numpy as np

class TreeTraverser:
    def __init__(self, num_threads):
        self.num_threads = num_threads

    def get_leaf_nodes(self, forest, data, oob_prediction):
        num_trees = len(forest.get_trees())
        leaf_nodes_by_tree = []

        thread_ranges = list(range(0, num_trees - 1, num_trees // self.num_threads)) + [num_trees]
        thread_ranges = [(thread_ranges[i], thread_ranges[i + 1]) for i in range(len(thread_ranges) - 1)]

        futures = []

        for i in range(len(thread_ranges) - 1):
            start_index, num_trees_batch = thread_ranges[i]
            futures.append(concurrent.futures.ThreadPoolExecutor.submit(self.get_leaf_node_batch, start_index, num_trees_batch, forest, data, oob_prediction))

        for future in concurrent.futures.as_completed(futures):
            leaf_nodes = future.result()
            leaf_nodes_by_tree.extend(leaf_nodes)

        return leaf_nodes_by_tree

    def get_valid_trees_by_sample(self, forest, data, oob_prediction):
        num_trees = len(forest.get_trees())
        num_samples = data.get_num_rows()

        result = [[True] * num_trees for _ in range(num_samples)]

        if oob_prediction:
            for tree_idx in range(num_trees):
                for sample in forest.get_trees()[tree_idx].get_drawn_samples():
                    result[sample][tree_idx] = False

        return result

    def get_leaf_node_batch(self, start, num_trees, forest, data, oob_prediction):
        num_samples = data.get_num_rows()
        all_leaf_nodes = []

        for i in range(num_trees):
            tree = forest.get_trees()[start + i]
            valid_samples = self.get_valid_samples(num_samples, tree, oob_prediction)
            leaf_nodes = tree.find_leaf_nodes(data, valid_samples)
            all_leaf_nodes.append(leaf_nodes)

        return all_leaf_nodes

    def get_valid_samples(self, num_samples, tree, oob_prediction):
        valid_samples = [True] * num_samples

        if oob_prediction:
            for sample in tree.get_drawn_samples():
                valid_samples[sample] = False

        return valid_samples

# Example usage:
# Initialize TreeTraverser with the desired number of threads, and then call get_leaf_nodes and get_valid_trees_by_sample.
# traverser = TreeTraverser(num_threads)
# leaf_nodes = traverser.get_leaf_nodes(forest, data, oob_prediction)
# valid_trees_by_sample = traverser.get_valid_trees_by_sample(forest, data, oob_prediction)
