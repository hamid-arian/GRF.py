import concurrent.futures
import math
from collections import defaultdict

class OptimizedPredictionCollector:
    def __init__(self, strategy, num_threads):
        self.strategy = strategy
        self.num_threads = num_threads

    def collect_predictions(self, forest, train_data, data, leaf_nodes_by_tree, valid_trees_by_sample, estimate_variance, estimate_error):
        num_samples = data.get_num_rows()
        thread_ranges = list(range(0, num_samples - 1, num_samples // self.num_threads)) + [num_samples]
        thread_ranges = [(thread_ranges[i], thread_ranges[i + 1]) for i in range(len(thread_ranges) - 1)]

        predictions = []

        def collect_predictions_batch(start, end):
            local_predictions = []

            for sample in range(start, end):
                average_value = []
                leaf_values = [] if estimate_variance or estimate_error else None

                # Create a list of weighted neighbors for this sample.
                num_leaves = 0
                for tree_index in range(len(forest.get_trees())):
                    if not valid_trees_by_sample[sample][tree_index]:
                        continue

                    leaf_nodes = leaf_nodes_by_tree[tree_index]
                    node = leaf_nodes[sample]

                    tree = forest.get_trees()[tree_index]
                    prediction_values = tree.get_prediction_values()

                    if not prediction_values.empty(node):
                        num_leaves += 1
                        self.add_prediction_values(node, prediction_values, average_value)
                        if leaf_values is not None:
                            leaf_values.append(prediction_values.get_values(node))

                # If this sample has no neighbors, then return placeholder predictions.
                if num_leaves == 0:
                    nan = [math.nan] * self.strategy.prediction_length()
                    nan_error = [math.nan]
                    local_predictions.append(Prediction(nan, nan if estimate_variance else [], nan_error, nan_error))
                    continue

                self.normalize_prediction_values(num_leaves, average_value)
                point_prediction = self.strategy.predict(average_value)

                prediction_values = PredictionValues(leaf_values, self.strategy.prediction_value_length())
                variance = self.strategy.compute_variance(average_value, prediction_values, forest.get_ci_group_size()) if estimate_variance else []

                mse = []
                mce = []

                if estimate_error:
                    error = self.strategy.compute_error(sample, average_value, prediction_values, data)
                    mse.append(error[0][0])
                    mce.append(error[0][1])

                prediction = Prediction(point_prediction, variance, mse, mce)

                self.validate_prediction(sample, prediction)
                local_predictions.append(prediction)

            return local_predictions

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(collect_predictions_batch, start, end) for start, end in thread_ranges]
            for future in concurrent.futures.as_completed(futures):
                predictions.extend(future.result())

        return predictions

    def add_prediction_values(self, node, prediction_values, combined_average):
        if not combined_average:
            combined_average.extend([0.0] * prediction_values.get_num_types())

        for type in range(prediction_values.get_num_types()):
            combined_average[type] += prediction_values.get(node, type)

    def normalize_prediction_values(self, num_leaves, combined_average):
        for i in range(len(combined_average)):
            combined_average[i] /= num_leaves

    def validate_prediction(self, sample, prediction):
        prediction_length = self.strategy.prediction_length()
        if len(prediction) != prediction_length:
            raise RuntimeError(f"Prediction for sample {sample} did not have the expected length.")

class Prediction:
    def __init__(self, point_prediction, variance, mse, mce):
        self.point_prediction = point_prediction
        self.variance = variance
        self.mse = mse
        self.mce = mce

class PredictionValues:
    def __init__(self, values, num_types):
        self.values = values
        self.num_types = num_types

    def empty(self, node):
        return not self.values or not self.values[node]

    def get(self, node, type):
        return self.values[node][type]

    def get_values(self, node):
        return self.values[node]

    def get_num_types(self):
        return self.num_types

# Example usage:
# Initialize OptimizedPredictionCollector and other required objects, and then call collect_predictions.
# collector = OptimizedPredictionCollector(strategy, num_threads)
# predictions = collector.collect_predictions(forest, train_data, data, leaf_nodes_by_tree, valid_trees_by_sample, True, True)
