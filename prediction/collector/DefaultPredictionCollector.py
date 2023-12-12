import concurrent.futures
import math
import numpy as np
from collections import defaultdict

class DefaultPredictionCollector:
    def __init__(self, strategy, num_threads):
        self.strategy = strategy
        self.num_threads = num_threads

    def collect_predictions(self, forest, train_data, data, leaf_nodes_by_tree, valid_trees_by_sample, estimate_variance, estimate_error):
        num_samples = data.shape[0]
        thread_ranges = list(range(0, num_samples - 1, num_samples // self.num_threads)) + [num_samples]
        thread_ranges = [(thread_ranges[i], thread_ranges[i + 1]) for i in range(len(thread_ranges) - 1)]

        predictions = []

        def collect_predictions_batch(start, end):
            local_predictions = []

            for sample in range(start, end):
                weights_by_sample = self.compute_weights(sample, forest, leaf_nodes_by_tree, valid_trees_by_sample)
                samples_by_tree = []

                if not weights_by_sample:
                    nan = [math.nan] * self.strategy.prediction_length()
                    local_predictions.append(Prediction(nan, nan if estimate_variance else [], [], []))
                    continue

                if estimate_variance:
                    samples_by_tree = [[] for _ in range(len(forest.get_trees()))]

                    for tree_index, valid in enumerate(valid_trees_by_sample[sample]):
                        if not valid:
                            continue
                        leaf_nodes = leaf_nodes_by_tree[tree_index]
                        node = leaf_nodes[sample]
                        tree = forest.get_trees()[tree_index]
                        leaf_samples = tree.get_leaf_samples()
                        samples_by_tree[node] = leaf_samples[node]

                point_prediction = self.strategy.predict(sample, weights_by_sample, train_data, data)
                variance = self.strategy.compute_variance(sample, samples_by_tree, weights_by_sample, train_data, data, forest.get_ci_group_size()) if estimate_variance else []

                if not point_prediction:
                    nan = [math.nan] * self.strategy.prediction_length()
                    local_predictions.append(Prediction(nan, nan if estimate_variance else [], [], []))
                else:
                    local_predictions.append(Prediction(point_prediction, variance, [], []))
                    self.validate_prediction(sample, point_prediction)

            return local_predictions

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(collect_predictions_batch, start, end) for start, end in thread_ranges]
            for future in concurrent.futures.as_completed(futures):
                predictions.extend(future.result())

        return predictions

    def compute_weights(self, sample, forest, leaf_nodes_by_tree, valid_trees_by_sample):
        weights_by_sample = defaultdict(float)

        for tree_index, valid in enumerate(valid_trees_by_sample[sample]):
            if not valid:
                continue
            leaf_nodes = leaf_nodes_by_tree[tree_index]
            node = leaf_nodes[sample]
            tree = forest.get_trees()[tree_index]
            weights_by_sample[node] += tree.get_weights()[sample]

        return weights_by_sample

    def validate_prediction(self, sample, prediction):
        prediction_length = self.strategy.prediction_length()
        if len(prediction) != prediction_length:
            raise RuntimeError(f"Prediction for sample {sample} did not have the expected length")

class Prediction:
    def __init__(self, point_prediction, variance, empty1, empty2):
        self.point_prediction = point_prediction
        self.variance = variance
        self.empty1 = empty1
        self.empty2 = empty2

# Example usage:
# Initialize DefaultPredictionCollector and other required objects, and then call collect_predictions.
# collector = DefaultPredictionCollector(strategy, num_threads)
# predictions = collector.collect_predictions(forest, train_data, data, leaf_nodes_by_tree, valid_trees_by_sample, True, True)
