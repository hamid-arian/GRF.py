# This file is part of generalized random forest (grf).
#
# grf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# grf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with grf. If not, see <http://www.gnu.org/licenses/>.

import numpy as np

class MultiCausalSplittingRule:
    def __init__(self, max_num_unique_values, min_node_size, alpha, imbalance_penalty,
                 response_length, num_treatments):
        self.min_node_size = min_node_size
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.response_length = response_length
        self.num_treatments = num_treatments
        self.counter = np.zeros(max_num_unique_values)
        self.weight_sums = np.zeros(max_num_unique_values)
        self.sums = np.zeros((max_num_unique_values, response_length))
        self.num_small_w = np.zeros((max_num_unique_values, num_treatments), dtype=int)
        self.sums_w = np.zeros((max_num_unique_values, num_treatments))
        self.sums_w_squared = np.zeros((max_num_unique_values, num_treatments))

    def find_best_split(self, data, node, possible_split_vars, responses_by_sample, samples,
                        split_vars, split_values, send_missing_left):
        num_samples = len(samples[node])

        # Precompute the sum of outcomes in this node.
        weight_sum_node = 0.0
        sum_node = np.zeros(self.response_length)
        sum_node_w = np.zeros(self.num_treatments)
        sum_node_w_squared = np.zeros(self.num_treatments)
        # Allocate W-array and re-use to avoid expensive copy-inducing calls to `data.get_treatments`
        treatments = np.zeros((num_samples, self.num_treatments))
        for i in range(num_samples):
            sample = samples[node][i]
            sample_weight = data.get_weight(sample)
            weight_sum_node += sample_weight
            sum_node += sample_weight * responses_by_sample[sample]
            treatments[i] = data.get_treatments(sample)

            sum_node_w += sample_weight * treatments[i]
            sum_node_w_squared += sample_weight * treatments[i] ** 2

        size_node = sum_node_w_squared - sum_node_w ** 2 / weight_sum_node
        min_child_size = size_node * self.alpha

        mean_w_node = sum_node_w / weight_sum_node
        num_node_small_w = np.sum(treatments < mean_w_node, axis=0)

        # Initialize the variables to track the best split variable.
        best_var = 0
        best_value = 0
        best_decrease = 0.0
        best_send_missing_left = True

        # For all possible split variables
        for var in possible_split_vars:
            self.find_best_split_value(data, node, var, num_samples, weight_sum_node, sum_node,
                                       mean_w_node, num_node_small_w, sum_node_w,
                                       sum_node_w_squared, min_child_size, treatments, best_value,
                                       best_var, best_decrease, best_send_missing_left,
                                       responses_by_sample, samples)

        # Stop if no good split found
        if best_decrease <= 0.0:
            return True

        # Save best values
        split_vars[node] = best_var
        split_values[node] = best_value
        send_missing_left[node] = best_send_missing_left
        return False

    def find_best_split_value(self, data, node, var, num_samples, weight_sum_node,
                              sum_node, mean_node_w, num_node_small_w, sum_node_w,
                              sum_node_w_squared, min_child_size, treatments,
                              best_value, best_var, best_decrease,
                              best_send_missing_left, responses_by_sample, samples):
        possible_split_values, sorted_samples, index = data.get_all_values(samples[node], var)

        # Try next variable if all equal for this
        if len(possible_split_values) < 2:
            return

        num_splits = len(possible_split_values) - 1

        self.counter[:num_splits] = 0
        self.weight_sums[:num_splits] = 0
        self.sums[:num_splits, :] = 0
        self.num_small_w[:num_splits, :] = 0
        self.sums_w[:num_splits, :] = 0
        self.sums_w_squared[:num_splits, :] = 0
        n_missing = 0
        weight_sum_missing = 0
        sum_missing = np.zeros(self.response_length)
        sum_w_missing = np.zeros(self.num_treatments)
        sum_w_squared_missing = np.zeros(self.num_treatments)
        num_small_w_missing = np.zeros(self.num_treatments)

        split_index = 0
        for i in range(num_samples - 1):
            sample = sorted_samples[i]
            next_sample = sorted_samples[i + 1]
            sort_index = index[i]
            sample_value = data.get(sample, var)
            sample_weight = data.get_weight(sample)

            if np.isnan(sample_value):
                weight_sum_missing += sample_weight
                sum_missing += sample_weight * responses_by_sample[sample]
                n_missing += 1

                sum_w_missing += sample_weight * treatments[sort_index]
                sum_w_squared_missing += sample_weight * treatments[sort_index] ** 2
                num_small_w_missing += (treatments[sort_index] < mean_node_w).astype(int)
            else:
                self.weight_sums[split_index] += sample_weight
                self.sums[split_index, :] += sample_weight * responses_by_sample[sample]
                self.counter[split_index] += 1

                self.sums_w[split_index, :] += sample_weight * treatments[sort_index]
                self.sums_w_squared[split_index, :] += sample_weight * treatments[sort_index] ** 2
                self.num_small_w[split_index, :] += (treatments[sort_index] < mean_node_w).astype(int)

            next_sample_value = data.get(next_sample, var)
            # if the next sample value is different, including the transition (..., NaN, Xij, ...)
            # then move on to the next bucket
            if sample_value != next_sample_value and not np.isnan(next_sample_value):
                split_index += 1

        n_left = n_missing
        weight_sum_left = weight_sum_missing
        sum_left = sum_missing
        sum_left_w = sum_w_missing
        sum_left_w_squared = sum_w_squared_missing
        num_left_small_w = num_small_w_missing

        # Compute decrease of impurity for each possible split
        for send_left in [True, False]:
            if not send_left:
                # A normal split with no NaNs, so we can stop early.
                if n_missing == 0:
                    break
                # It is not necessary to adjust n_right or sum_right as the missing
                # part is included in the total sum.
                n_left = 0
                weight_sum_left = 0
                sum_left = np.zeros(self.response_length)
                sum_left_w = np.zeros(self.num_treatments)
                sum_left_w_squared = np.zeros(self.num_treatments)
                num_left_small_w = np.zeros(self.num_treatments)

            for i in range(num_splits):
                # not necessary to evaluate sending right when splitting on NaN.
                if i == 0 and not send_left:
                    continue

                n_left += self.counter[i]
                weight_sum_left += self.weight_sums[i]
                num_left_small_w += self.num_small_w[i, :]
                sum_left += self.sums[i, :]
                sum_left_w += self.sums_w[i, :]
                sum_left_w_squared += self.sums_w_squared[i, :]

                # Skip this split if the left child does not contain enough
                # w values below and above the parent's mean.
                if (num_left_small_w < self.min_node_size).any() or \
                        (n_left - num_left_small_w < self.min_node_size).any():
                    continue

                # Stop if the right child does not contain enough w values below
                # and above the parent's mean.
                n_right = num_samples - n_left
                if (num_node_small_w - num_left_small_w < self.min_node_size).any() or \
                        (n_right - num_node_small_w + num_left_small_w < self.min_node_size).any():
                    break

                # Calculate relevant quantities for the left child.
                size_left = sum_left_w_squared - sum_left_w ** 2 / weight_sum_left
                # Skip this split if the left child's variance is too small.
                if (size_left < min_child_size).any() or \
                        (self.imbalance_penalty > 0.0 and (size_left == 0).all()):
                    continue

                # Calculate relevant quantities for the right child.
                weight_sum_right = weight_sum_node - weight_sum_left
                size_right = (sum_node_w_squared - sum_left_w_squared -
                              (sum_node_w - sum_left_w) ** 2 / weight_sum_right)
                # Skip this split if the right child's variance is too small.
                if (size_right < min_child_size).any() or \
                        (self.imbalance_penalty > 0.0 and (size_right == 0).all()):
                    continue

                # Calculate the decrease in impurity.
                decrease = (np.sum(sum_left ** 2) / weight_sum_left +
                            np.sum((sum_node - sum_left) ** 2) / weight_sum_right)

                # Penalize splits that are too close to the edges of the data.
                penalty = self.imbalance_penalty * (1.0 / n_left + 1.0 / n_right)
                decrease -= penalty

                # If better than before, use this
                if decrease > best_decrease:
                    best_value = possible_split_values[i]
                    best_var = var
                    best_decrease = decrease
                    best_send_missing_left = send_left

        return


# Example usage:
# You need to replace the following placeholders with actual implementations
# of Data and other dependencies.

# class Data:
#     def get_weight(self, sample):
#         pass
#
#     def get_treatments(self, sample):
#         pass
#
#     def get(self, sample, var):
#         pass
#
#     def get_all_values(self, samples, var):
#         pass
