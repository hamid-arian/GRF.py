import numpy as np


class ProbabilitySplittingRule:
    def __init__(self, max_num_unique_values, num_classes, alpha, imbalance_penalty):
        self.num_classes = num_classes
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.counter = np.zeros(max_num_unique_values, dtype=int)
        self.counter_per_class = np.zeros((max_num_unique_values, num_classes), dtype=float)

    def find_best_split(self, data, node, possible_split_vars, responses_by_sample, samples,
                        split_vars, split_values, send_missing_left):
        size_node = len(samples[node])
        min_child_size = max(int(np.ceil(size_node * self.alpha)), 1)

        class_counts = np.zeros(self.num_classes, dtype=float)
        for i in range(size_node):
            sample = samples[node][i]
            sample_class = round(responses_by_sample[sample, 0])
            sample_weight = data.get_weight(sample)
            class_counts[sample_class] += sample_weight

        # Initialize the variables to track the best split variable.
        best_var = 0
        best_value = 0
        best_decrease = 0.0
        best_send_missing_left = True

        # For all possible split variables
        for var in possible_split_vars:
            self.find_best_split_value(data, node, var, self.num_classes, class_counts, size_node,
                                       min_child_size, best_value, best_var, best_decrease,
                                       best_send_missing_left, responses_by_sample, samples)

        # Stop if no good split found
        if best_decrease <= 0.0:
            return True

        # Save best values
        split_vars[node] = best_var
        split_values[node] = best_value
        send_missing_left[node] = best_send_missing_left
        return False

    def find_best_split_value(self, data, node, var, num_classes, class_counts,
                              size_node, min_child_size, best_value, best_var,
                              best_decrease, best_send_missing_left,
                              responses_by_sample, samples):
        possible_split_values, sorted_samples = data.get_all_values(samples[node], var)

        # Try next variable if all equal for this
        if len(possible_split_values) < 2:
            return

        num_splits = len(possible_split_values) - 1

        self.counter_per_class[:num_splits, :] = 0
        self.counter[:num_splits] = 0
        n_missing = 0
        class_counts_missing = np.zeros(self.num_classes, dtype=float)

        split_index = 0
        for i in range(size_node - 1):
            sample = sorted_samples[i]
            next_sample = sorted_samples[i + 1]
            sample_value = data.get(sample, var)
            sample_class = int(responses_by_sample[sample, 0])
            sample_weight = data.get_weight(sample)

            if np.isnan(sample_value):
                class_counts_missing[sample_class] += sample_weight
                n_missing += 1
            else:
                self.counter[split_index] += 1
                self.counter_per_class[split_index, sample_class] += sample_weight

            next_sample_value = data.get(next_sample, var)
            # if the next sample value is different, move on to the next bucket
            if sample_value != next_sample_value and not np.isnan(next_sample_value):
                split_index += 1

        n_left = n_missing
        class_counts_left = class_counts_missing

        # Compute decrease of impurity for each possible split
        for send_left in [True, False]:
            if not send_left:
                # A normal split with no NaNs, so we can stop early.
                if n_missing == 0:
                    break
                # It is not necessary to adjust n_right or sum_right as the missing
                # part is included in the total sum.
                n_left = 0
                class_counts_left = np.zeros(self.num_classes, dtype=float)

            for i in range(num_splits):
                # not necessary to evaluate sending right when splitting on NaN.
                if i == 0 and not send_left:
                    continue

                n_left += self.counter[i]

                # Stop if the right child is too small.
                n_right = size_node - n_left
                if n_right < min_child_size:
                    break

                # Sum of squares
                sum_left = np.sum(class_counts_left ** 2)
                sum_right = np.sum((class_counts - class_counts_left) ** 2)

                # Skip to the next value if the left child is too small.
                if n_left < min_child_size:
                    continue

                # Decrease of impurity
                decrease = sum_right / n_right + sum_left / n_left

                # Penalize splits that are too close to the edges of the data.
                penalty = self.imbalance_penalty * (1.0 / n_left + 1.0 / n_right)
                decrease -= penalty

                # If better than before, use this
                if decrease > best_decrease:
                    best_value = possible_split_values[i]
                    best_var = var
                    best_decrease = decrease
                    best_send_missing_left = send_left
