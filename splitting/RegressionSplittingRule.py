import numpy as np


class RegressionSplittingRule:
    def __init__(self, max_num_unique_values, alpha, imbalance_penalty):
        """
        Initialize a RegressionSplittingRule.

        :param max_num_unique_values: The maximum number of unique values for a variable.
        :param alpha: Minimum node size proportion as fraction of total samples.
        :param imbalance_penalty: Penalty for imbalanced splits.
        """
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.counter = np.zeros(max_num_unique_values, dtype=int)
        self.sums = np.zeros(max_num_unique_values)
        self.weight_sums = np.zeros(max_num_unique_values)

    def find_best_split(self, data, node, possible_split_vars, responses_by_sample, samples):
        """
        Find the best split for a node.

        :param data: The data used for training the tree.
        :param node: The index of the node to be split.
        :param possible_split_vars: A list of variables considered for splitting.
        :param responses_by_sample: The responses associated with each sample.
        :param samples: A list of samples at each node.
        :return: True if no split is found, False otherwise.
        """
        size_node = len(samples[node])
        min_child_size = max(int(np.ceil(size_node * self.alpha)), 1)

        sum_node = weight_sum_node = 0.0
        for sample in samples[node]:
            sample_weight = data.get_weight(sample)
            weight_sum_node += sample_weight
            sum_node += sample_weight * responses_by_sample[sample, 0]

        best_var, best_value, best_decrease, best_send_missing_left = 0, 0.0, 0.0, True

        for var in possible_split_vars:
            self.find_best_split_value(data, node, var, weight_sum_node, sum_node, size_node, min_child_size, best_value, best_var, best_decrease, best_send_missing_left, responses_by_sample, samples)

        if best_decrease <= 0.0:
            return True

        split_vars[node] = best_var
        split_values[node] = best_value
        send_missing_left[node] = best_send_missing_left

        return False


    def find_best_split_value(self, data, node, var, weight_sum_node, sum_node, size_node, min_child_size, best_value, best_var, best_decrease, best_send_missing_left, responses_by_sample, samples):
        possible_split_values, sorted_samples = data.get_all_values(samples[node], var)

        if len(possible_split_values) < 2:
            return

        num_splits = len(possible_split_values) - 1
        self.weight_sums[:num_splits] = 0
        self.counter[:num_splits] = 0
        self.sums[:num_splits] = 0
        n_missing = weight_sum_missing = sum_missing = 0

        split_index = 0
        for i in range(size_node - 1):
            sample = sorted_samples[i]
            next_sample = sorted_samples[i + 1]
            sample_value = data.get(sample, var)
            response = responses_by_sample[sample, 0]
            sample_weight = data.get_weight(sample)

            if np.isnan(sample_value):
                weight_sum_missing += sample_weight
                sum_missing += sample_weight * response
                n_missing += 1
            else:
                self.weight_sums[split_index] += sample_weight
                self.sums[split_index] += sample_weight * response
                self.counter[split_index] += 1

            next_sample_value = data.get(next_sample, var)
            if sample_value != next_sample_value and not np.isnan(next_sample_value):
                split_index += 1

        n_left = weight_sum_left = sum_left = 0

        for send_left in [True, False]:
            if not send_left and n_missing == 0:
                break

            n_left = weight_sum_left = sum_left = 0

            for i in range(num_splits):
                if i == 0 and not send_left:
                    continue

                n_left += self.counter[i]
                weight_sum_left += self.weight_sums[i]
                sum_left += self.sums[i]

                if n_left < min_child_size:
                    continue

                n_right = size_node - n_left
                if n_right < min_child_size:
                    break

                weight_sum_right = weight_sum_node - weight_sum_left
                sum_right = sum_node - sum_left
                decrease = sum_left**2 / weight_sum_left + sum_right**2 / weight_sum_right
                penalty = self.imbalance_penalty * (1.0 / n_left + 1.0 / n_right)
                decrease -= penalty

                if decrease > best_decrease:
                    best_value = possible_split_values[i]
                    best_var = var
                    best_decrease = decrease
                    best_send_missing_left = send_left
