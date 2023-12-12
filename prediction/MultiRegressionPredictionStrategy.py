import numpy as np

class MultiRegressionPredictionStrategy:
    def __init__(self, num_outcomes):
        self.num_outcomes = num_outcomes
        self.num_types = 1 + num_outcomes
        self.weight_index = num_outcomes

    def prediction_length(self):
        return self.num_outcomes

    def predict(self, average):
        predictions = []
        predictions.reserve(self.num_outcomes)
        weight_bar = average[self.weight_index]
        for j in range(self.num_outcomes):
            predictions.append(average[j] / weight_bar)

        return predictions

    def compute_variance(self, average, leaf_values, ci_group_size):
        return [0.0]

    def prediction_value_length(self):
        return self.num_types

    def precompute_prediction_values(self, leaf_samples, data):
        num_leaves = len(leaf_samples)
        values = []

        for i in range(num_leaves):
            leaf_node = leaf_samples[i]
            num_samples = len(leaf_node)
            if num_samples == 0:
                continue

            sum_outcomes = np.zeros(self.num_outcomes)
            sum_weight = 0.0

            for sample in leaf_node:
                weight = data.get_weight(sample)
                sum_outcomes += weight * data.get_outcomes(sample)
                sum_weight += weight

            # if total weight is very small, treat the leaf as empty
            if np.abs(sum_weight) <= 1e-16:
                continue

            # store sufficient statistics in order
            # {outcome_1, ..., outcome_M, weight_sum}
            value = []
            value.reserve(self.num_types)
            for j in range(self.num_outcomes):
                value.append(sum_outcomes[j] / num_samples)
            value.append(sum_weight / num_samples)
            values.append(value)

        return PredictionValues(values, self.num_types)

    def compute_error(self, sample, average, leaf_values, data):
        return [(np.nan, np.nan)]
