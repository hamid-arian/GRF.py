import numpy as np

class CausalSurvivalPredictionStrategy:
    NUMERATOR = 0
    DENOMINATOR = 1
    NUM_TYPES = 2

    def prediction_length(self):
        return 1

    def predict(self, average):
        return [average[self.NUMERATOR] / average[self.DENOMINATOR]]

    def compute_variance(self, average, leaf_values, ci_group_size):
        v_est = average[self.DENOMINATOR]
        average_tau = average[self.NUMERATOR] / average[self.DENOMINATOR]

        num_good_groups = 0
        psi_squared = 0
        psi_grouped_squared = 0

        for group in range(len(leaf_values) // ci_group_size):
            good_group = all(not leaf_values.empty(group * ci_group_size + j) for j in range(ci_group_size))
            if not good_group:
                continue

            num_good_groups += 1
            group_psi = 0

            for j in range(ci_group_size):
                i = group * ci_group_size + j
                leaf_value = leaf_values.get_values(i)

                psi_1 = leaf_value[self.NUMERATOR] - leaf_value[self.DENOMINATOR] * average_tau
                psi_squared += psi_1 * psi_1
                group_psi += psi_1

            group_psi /= ci_group_size
            psi_grouped_squared += group_psi * group_psi

        var_between = psi_grouped_squared / num_good_groups
        var_total = psi_squared / (num_good_groups * ci_group_size)
        group_noise = (var_total - var_between) / (ci_group_size - 1)
        var_debiased = self.bayes_debiaser.debias(var_between, group_noise, num_good_groups)

        variance_estimate = var_debiased / (v_est * v_est)
        return [variance_estimate]

    def prediction_value_length(self):
        return self.NUM_TYPES

    def precompute_prediction_values(self, leaf_samples, data):
        num_leaves = len(leaf_samples)
        values = []

        for i in range(num_leaves):
            leaf_node = leaf_samples[i]
            leaf_size = len(leaf_node)
            if leaf_size == 0:
                continue

            numerator_sum = 0
            denominator_sum = 0
            sum_weight = 0

            for sample in leaf_node:
                weight = data.get_weight(sample)
                numerator_sum += weight * data.get_causal_survival_numerator(sample)
                denominator_sum += weight * data.get_causal_survival_denominator(sample)
                sum_weight += weight

            if np.abs(sum_weight) <= 1e-16:
                continue

            value = [0] * self.NUM_TYPES
            value[self.NUMERATOR] = numerator_sum / leaf_size
            value[self.DENOMINATOR] = denominator_sum / leaf_size
            values.append(value)

        return PredictionValues(values, self.NUM_TYPES)

    def compute_error(self, sample, average, leaf_values, data):
        return [(np.nan, np.nan)]
