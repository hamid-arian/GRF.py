import numpy as np

class InstrumentalSplittingRule:
    def __init__(self, max_num_unique_values, min_node_size, alpha, imbalance_penalty):
        self.min_node_size = min_node_size
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty

        self.counter = np.zeros(max_num_unique_values, dtype=int)
        self.weight_sums = np.zeros(max_num_unique_values, dtype=float)
        self.sums = np.zeros(max_num_unique_values, dtype=float)
        self.num_small_z = np.zeros(max_num_unique_values, dtype=int)
        self.sums_z = np.zeros(max_num_unique_values, dtype=float)
        self.sums_z_squared = np.zeros(max_num_unique_values, dtype=float)

    def find_best_split(self, data, node, possible_split_vars, responses_by_sample, samples, split_vars, split_values, send_missing_left):
        num_samples = len(samples[node])

        # Precompute relevant quantities for this node.
        weight_sum_node = 0.0
        sum_node = 0.0
        sum_node_z = 0.0
        sum_node_z_squared = 0.0
        for sample in samples[node]:
            sample_weight = data.get_weight(sample)
            weight_sum_node += sample_weight
            sum_node += sample_weight * responses_by_sample[sample, 0]

            z = data.get_instrument(sample)
            sum_node_z += sample_weight * z
            sum_node_z_squared += sample_weight * z * z

        size_node = sum_node_z_squared - sum_node_z * sum_node_z / weight_sum_node
        min_child_size = size_node * self.alpha

        mean_z_node = sum_node_z / weight_sum_node
        num_node_small_z = sum(1 for sample in samples[node] if data.get_instrument(sample) < mean_z_node)

        # Initialize variables to track the best split variable.
        best_var, best_value, best_decrease, best_send_missing_left = 0, 0, 0.0, True

        for var in possible_split_vars:
            self.find_best_split_value(data, node, var, num_samples, weight_sum_node, sum_node, mean_z_node,
                                       num_node_small_z, sum_node_z, sum_node_z_squared, min_child_size, best_value,
                                       best_var, best_decrease, best_send_missing_left, responses_by_sample, samples)

        # Stop if no good split found
        if best_decrease <= 0.0:
            return True

        # Save best values
        split_vars[node] = best_var
        split_values[node] = best_value
        send_missing_left[node] = best_send_missing_left
        return False

    def find_best_split_value(self, data, node, var, num_samples, weight_sum_node, sum_node, mean_node_z,
                              num_node_small_z, sum_node_z, sum_node_z_squared, min_child_size, best_value,
                              best_var, best_decrease, best_send_missing_left, responses_by_sample, samples):
        possible_split_values, sorted_samples = data.get_all_values(samples[node], var)

        # Try next variable if all equal for this
        if len(possible_split_values) < 2:
            return

        num_splits = len(possible_split_values) - 1

        self.counter.fill(0)
        self.weight_sums.fill(0.0)
        self.sums.fill(0.0)
        self.num_small_z.fill(0)
        self.sums_z.fill(0.0)
        self.sums_z_squared.fill(0.0)
        n_missing = 0
        weight_sum_missing = 0
        sum_missing = 0
        sum_z_missing = 0
        sum_z_squared_missing = 0
        num_small_z_missing = 0

        split_index = 0
        for i in range(num_samples - 1):
            sample = sorted_samples[i]
            next_sample = sorted_samples[i + 1]
            sample_value = data.get(sample, var)
            z = data.get_instrument(sample)
            sample_weight = data.get_weight(sample)

            if np.isnan(sample_value):
                weight_sum_missing += sample_weight
                sum_missing += sample_weight * responses_by_sample[sample, 0]
                n_missing += 1

                sum_z_missing += sample_weight * z
                sum_z_squared_missing += sample_weight * z * z
                if z < mean_node_z:
                    num_small_z_missing += 1
            else:
                self.weight_sums[split_index] += sample_weight
                self.sums[split_index] += sample_weight * responses_by_sample[sample, 0]
                self.counter[split_index] += 1

                self.sums_z[split_index] += sample_weight * z
                self.sums_z_squared[split_index] += sample_weight * z * z
                if z < mean_node_z:
                    self.num_small_z[split_index] += 1

            next_sample_value = data.get(next_sample, var)
            # if the next sample value is different, including the transition (..., NaN, Xij, ...)
            # then move on to the next bucket (all logical operators with NaN evaluates to false by default)
            if sample_value != next_sample_value and not np.isnan(next_sample_value):
                split_index += 1

        n_left = n_missing
        weight_sum_left = weight_sum_missing
        sum_left = sum_missing
        sum_left_z = sum_z_missing
        sum_left_z_squared = sum_z_squared_missing
        num_left_small_z = num_small_z_missing

        # Compute decrease of impurity for each possible split.
        for send_left in [True, False]:
            if not send_left:
                # A normal split with no NaNs, so we can stop early.
                if n_missing == 0:
                    break
                # It is not necessary to adjust n_right or sum_right as the missing
                # part is included in the total sum.
                n_left = 0
                weight_sum_left = 0
                sum_left = 0
                sum_left_z = 0
                sum_left_z_squared = 0
                num_left_small_z = 0

            for i in range(num_splits):
                # not necessary to evaluate sending right when splitting on NaN.
                if i == 0 and not send_left:
                    continue

                n_left += self.counter[i]
                num_left_small_z += self.num_small_z[i]
                weight_sum_left += self.weight_sums[i]
                sum_left += self.sums[i]
                sum_left_z += self.sums_z[i]
                sum_left_z_squared += self.sums_z_squared[i]

                # Skip this split if the left child does not contain enough
                # z values below and above the parent's mean.
                num_left_large_z = n_left - num_left_small_z
                if num_left_small_z < self.min_node_size or num_left_large_z < self.min_node_size:
                    continue

                # Stop if the right child does not contain enough z values below
                # and above the parent's mean.
                num_failures_right = num_node_small_z - num_left_small_z
                if num_failures_right < self.min_node_size:
                    break

                # Calculate relevant quantities for the left child.
                size_left = sum_left_z_squared - sum_left_z * sum_left_z / weight_sum_left
                # Skip this split if the left child's variance is too small.
                if size_left < min_child_size or (self.imbalance_penalty > 0.0 and size_left == 0):
                    continue

                # Calculate relevant quantities for the right child.
                weight_sum_right = weight_sum_node - weight_sum_left
                sum_right = sum_node - sum_left
                sum_right_z_squared = sum_node_z_squared - sum_left_z_squared
                sum_right_z = sum_node_z - sum_left_z
                size_right = sum_right_z_squared - sum_right_z * sum_right_z / weight_sum_right

                # Skip this split if the right child's variance is too small.
                if size_right < min_child_size or (self.imbalance_penalty > 0.0 and size_right == 0):
                    continue

                # Calculate the decrease in impurity.
                decrease = (sum_left * sum_left / weight_sum_left) + (sum_right * sum_right / weight_sum_right)
                # Penalize splits that are too close to the edges of the data.
                decrease -= self.imbalance_penalty * (1.0 / size_left + 1.0 / size_right)

                # Save this split if it is the best seen so far.
                if decrease > best_decrease:
                    best_value = possible_split_values[i]
                    best_var = var
                    best_decrease = decrease
                    best_send_missing_left = send_left
