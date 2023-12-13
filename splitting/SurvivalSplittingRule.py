import numpy as np


class SurvivalSplittingRule:
    def __init__(self, alpha):
        self.alpha = alpha

    def find_best_split(self, data, node, possible_split_vars, responses_by_sample,
                        samples_by_node, split_vars, split_values, send_missing_left):
        samples = samples_by_node[node]

        # The splitting rule output
        best_value = 0
        best_var = 0
        best_send_missing_left = True
        best_logrank = 0

        self.find_best_split_internal(data, possible_split_vars, responses_by_sample, samples,
                                      best_value, best_var, best_send_missing_left, best_logrank)

        # Stop if no good split found
        if best_logrank <= 0.0:
            return True

        # Save best values
        split_vars[node] = best_var
        split_values[node] = best_value
        send_missing_left[node] = best_send_missing_left
        return False

    def find_best_split_internal(self, data, possible_split_vars, responses_by_sample, samples,
                                 best_value, best_var, best_send_missing_left, best_logrank):
        size_node = len(samples)
        min_child_size = max(int(np.ceil(size_node * self.alpha)), 1)

        # Get the failure values t1, ..., tm in this node
        failure_values = [responses_by_sample[sample, 0] for sample in samples if data.is_failure(sample)]

        num_failures_node = len(failure_values)
        failure_values.sort()
        failure_values = list(dict.fromkeys(failure_values))  # Remove duplicates

        # The number of unique failure values in this node
        num_failures = len(failure_values)
        # If there are no failures or only one failure time, there is nothing to do.
        if num_failures <= 1:
            return

        # The number of failures at each time in the parent node. Entry 0 will be zero.
        # (Entry 0 is for time k < t1)
        count_failure = [0] * (num_failures + 1)
        # The number of censored observations at each time in the parent node.
        count_censor = [0] * (num_failures + 1)
        # The number of samples in the parent node at risk at each time point, i.e. the count of observations
        # with observed time greater than or equal to the given failure time. Entry 0 will be equal to the number
        # of samples (and the entries will always be monotonically decreasing)
        at_risk = [0] * (num_failures + 1)
        at_risk[0] = float(size_node)

        # Allocating an N-sized (full data set size) array is faster than a hash table
        relabeled_failures = [0] * data.get_num_rows()

        numerator_weights = [0] * (num_failures + 1)
        denominator_weights = [0] * (num_failures + 1)

        # Relabel the failure values to range from 0 to the number of failures in this node
        for sample in samples:
            failure_value = responses_by_sample[sample, 0]
            new_failure_value = len([fv for fv in failure_values if fv <= failure_value])
            relabeled_failures[sample] = new_failure_value
            if data.is_failure(sample):
                count_failure[new_failure_value] += 1
            else:
                count_censor[new_failure_value] += 1

        for time in range(1, num_failures + 1):
            at_risk[time] = at_risk[time - 1] - count_failure[time - 1] - count_censor[time - 1]

            # The logrank statistic is (using the notation in Ishwaran et al. (2008))
            # sum over all k: dk,l - Yk,l * dk/Yk divided by:
            #  Yk,l / Yk * (1 - Yk,l / Yk) * (Yk - dk) / (Yk - 1) dk
            # All terms involving only Yk or dk remain unchanged for each split
            # and can be precomputed here.
            Yk = at_risk[time]
            dk = count_failure[time]
            numerator_weights[time] = dk / Yk
            denominator_weights[time] = (Yk - dk) / (Yk - 1) * dk / (Yk * Yk)

        for var in possible_split_vars:
            self.find_best_split_value(data, var, size_node, min_child_size, num_failures_node, num_failures,
                                       best_value, best_var, best_logrank, best_send_missing_left, samples,
                                       relabeled_failures, count_failure, at_risk, numerator_weights, denominator_weights)

    def find_best_split_value(self, data, var, size_node, min_child_size, num_failures_node, num_failures,
                              best_value, best_var, best_logrank, best_send_missing_left, samples, relabeled_failures,
                              count_failure, count_censor, at_risk, numerator_weights, denominator_weights):
        # possible_split_values contains all the unique split values for this variable in increasing order
        # sorted_samples contain the samples in this node in increasing order
        # if there are missing values, these are placed first
        # (if all Xij's are continuous, these two vectors have the same length)
        possible_split_values, sorted_samples = data.get_all_values(samples, var)

        # Try next variable if all equal for this
        if len(possible_split_values) < 2:
            return

        left_count_failure = [0] * (num_failures + 1)
        left_count_censor = [0] * (num_failures + 1)
        cum_sums = [0] * (num_failures + 1)
        n_missing = 0
        num_failures_missing = 0

        # Loop through all samples to scan for missing values
        start_sample = n_missing = n_missing if possible_split_values[0] is None else 0
        for i in range(start_sample, size_node - 1):
            sample = sorted_samples[i]
            sample_value = data.get(sample, var)
            sample_time = relabeled_failures[sample]

            if np.isnan(sample_value):
                if data.is_failure(sample):
                    left_count_failure[sample_time] += 1
                    num_failures_missing += 1
                else:
                    left_count_censor[sample_time] += 1
                n_missing += 1

        num_splits = len(possible_split_values) - 1
        n_left = num_failures_left = split_index = 0
        start_sample = n_missing - 1 if n_missing > 0 else 0

        for send_left in [True, False]:
            if not send_left:
                # A normal split with no NaNs, so we can stop early.
                if n_missing == 0:
                    break
                # Else, send all missing right
                left_count_failure = [0] * (num_failures + 1)
                left_count_censor = [0] * (num_failures + 1)
                n_left = 0
                num_failures_left = 0
                # Not necessary to evaluate splitting on NaN when sending right.
                split_index = 1
                start_sample = n_missing

            for i in range(start_sample, size_node - 1):
                sample = sorted_samples[i]
                next_sample = sorted_samples[i + 1]
                sample_value = data.get(sample, var)
                next_sample_value = data.get(next_sample, var)
                sample_time = relabeled_failures[sample]

                # If there are missing values, we evaluate splitting on NaN when send_left is true
                # and i = n_missing - 1, which is why we need to check for missing below.
                split_on_missing = np.isnan(sample_value)

                if not split_on_missing:
                    n_left += 1

                if not split_on_missing:
                    if data.is_failure(sample):
                        left_count_failure[sample_time] += 1
                        num_failures_left += 1
                    else:
                        left_count_censor[sample_time] += 1

                # Skip this split if one child is too small (i.e., too few failures)
                if num_failures_left < min_child_size:
                    if sample_value != next_sample_value:
                        split_index += 1
                    continue

                # Stop if the right child is too small.
                num_failures_right = num_failures_node - num_failures_left
                if num_failures_right < min_child_size:
                    break

                # If the next sample value is different, we can evaluate a split here
                if sample_value != next_sample_value:
                    logrank = self.compute_logrank(num_failures, n_left, cum_sums, left_count_failure,
                                                   left_count_censor, at_risk, numerator_weights, denominator_weights)
                    if logrank > best_logrank:
                        best_value = possible_split_values[split_index]
                        best_var = var
                        best_logrank = logrank
                        best_send_missing_left = send_left
                    split_index += 1

                if split_index == num_splits:
                    break

    def compute_logrank(self, num_failures, n_left, cum_sums, left_count_failure, left_count_censor, at_risk,
                        numerator_weights, denominator_weights):
        numerator = denominator = logrank = 0
        for time in range(1, num_failures + 1):
            cum_sums[time] = cum_sums[time - 1] + left_count_failure[time - 1] + left_count_censor[time - 1]
            Yl = n_left - cum_sums[time]
            if Yl == 0:
                break
            Y = at_risk[time]
            # The logrank denominator requires at least two at risk
            if Y < 2:
                break
            dl = left_count_failure[time]
            numerator = numerator + dl - Yl * numerator_weights[time]
            denominator = denominator + Yl * (Y - Yl) * denominator_weights[time]

        if denominator > 0:
            logrank = numerator * numerator / denominator

        return logrank
