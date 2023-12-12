import numpy as np

class MultiCausalPredictionStrategy:
    def __init__(self, num_treatments, num_outcomes):
        self.num_treatments = num_treatments
        self.num_outcomes = num_outcomes
        self.num_types = num_treatments * (num_treatments + num_outcomes + 1) + num_outcomes + 1
        self.weight_index = 0
        self.Y_index = 1
        self.W_index = self.Y_index + num_outcomes
        self.YW_index = self.W_index + num_treatments
        self.WW_index = self.YW_index + num_treatments * num_outcomes
        self.bayes_debiaser = ObjectiveBayesDebiaser()

    def prediction_length(self):
        return self.num_treatments * self.num_outcomes

    def predict(self, average):
        # Re-construct the relevant data structures from the list produced in precompute_prediction_values
        weight_bar = average[self.weight_index]
        Y_bar = np.array(average[self.Y_index:self.Y_index + self.num_outcomes])
        W_bar = np.array(average[self.W_index:self.W_index + self.num_treatments])
        YW_bar = np.array(average[self.YW_index:self.YW_index + self.num_treatments * self.num_outcomes]).reshape((self.num_treatments, self.num_outcomes))
        WW_bar = np.array(average[self.WW_index:self.WW_index + self.num_treatments * self.num_treatments]).reshape((self.num_treatments, self.num_treatments))

        # Why we do not do a `...ldlt().solve(...)` instead of inverse: because dim(W) is low (intended use-case)
        # and ^-1 replicates the behavior of InstrumentalPredictionStrategy for dim(W) = 1.
        theta = np.linalg.inv(WW_bar * weight_bar - np.outer(W_bar, W_bar)) @ (YW_bar * weight_bar - np.outer(W_bar, Y_bar))

        return list(theta.flatten())

    def compute_variance(self, average, leaf_values, ci_group_size):
        if self.num_outcomes > 1:
            raise RuntimeError("Pointwise variance estimates are only implemented for one outcome.")

        weight_bar = average[self.weight_index]
        Y_bar = average[self.Y_index]
        W_bar = np.array(average[self.W_index:self.W_index + self.num_treatments])
        YW_bar = np.array(average[self.YW_index:self.YW_index + self.num_treatments * self.num_outcomes]).reshape((self.num_treatments, self.num_outcomes))
        WW_bar = np.array(average[self.WW_index:self.WW_index + self.num_treatments * self.num_treatments]).reshape((self.num_treatments, self.num_treatments))

        theta = np.linalg.inv(WW_bar * weight_bar - np.outer(W_bar, W_bar)) @ (YW_bar * weight_bar - Y_bar * W_bar)
        main_effect = (Y_bar - theta @ W_bar) / weight_bar

        # NOTE: could potentially use pseudoinverses.
        k = weight_bar - np.transpose(W_bar) @ np.linalg.inv(WW_bar) @ W_bar
        term1 = np.linalg.inv(WW_bar) + 1 / k * np.linalg.inv(WW_bar) @ np.outer(W_bar, W_bar) @ np.linalg.inv(WW_bar)
        term2 = 1 / k * np.linalg.inv(WW_bar) @ W_bar

        num_good_groups = 0
        rho_squared = np.zeros(self.num_treatments)
        rho_grouped_squared = np.zeros(self.num_treatments)

        group_rho = np.zeros(self.num_treatments)
        psi_1 = np.zeros(self.num_treatments)
        rho = np.zeros(self.num_treatments)

        for group in range(leaf_values.get_num_nodes() // ci_group_size):
            good_group = all(not leaf_values.empty(group * ci_group_size + j) for j in range(ci_group_size))
            if not good_group:
                continue

            num_good_groups += 1
            group_rho.fill(0)
            for j in range(ci_group_size):
                i = group * ci_group_size + j
                leaf_value = leaf_values.get_values(i)
                leaf_weight = leaf_value[self.weight_index]
                leaf_Y = leaf_value[self.Y_index]
                leaf_W = np.array(leaf_value[self.W_index:self.W_index + self.num_treatments])
                leaf_YW = np.array(leaf_value[self.YW_index:self.YW_index + self.num_treatments * self.num_outcomes]).reshape((self.num_treatments, self.num_outcomes))
                leaf_WW = np.array(leaf_value[self.WW_index:self.WW_index + self.num_treatments * self.num_treatments]).reshape((self.num_treatments, self.num_treatments))

                psi_1 = leaf_YW - leaf_WW @ theta - leaf_W @ main_effect
                psi_2 = leaf_Y - np.transpose(leaf_W) @ theta - leaf_weight * main_effect

                rho = term1 @ psi_1 - term2 * psi_2
                rho_squared += rho ** 2
                group_rho += rho

            group_rho /= ci_group_size
            rho_grouped_squared += group_rho ** 2

        var_between = rho_grouped_squared / num_good_groups
        var_total = rho_squared / (num_good_groups * ci_group_size)

        # This is the amount by which var_between is inflated due to using small groups
        group_noise = (var_total - var_between) / (ci_group_size - 1)

        # A simple variance correction, would be to use:
        # var_debiased = var_between - group_noise.
        # However, this may be biased in small samples; we do an elementwise objective
        # Bayes analysis of variance instead to avoid negative values.
        var_debiased = np.zeros(self.num_treatments)
        for i in range(self.num_treatments):
            var_debiased[i] = self.bayes_debiaser.debias(var_between[i], group_noise[i], num_good_groups)

        return var_debiased

    def prediction_value_length(self):
        return self.num_types

    def precompute_prediction_values(self, leaf_samples, data):
        num_leaves = len(leaf_samples)
        values = []

        for i in range(num_leaves):
            num_samples = len(leaf_samples[i])
            if num_samples == 0:
                continue

            sum_Y = np.zeros(self.num_outcomes)
            sum_W = np.zeros(self.num_treatments)
            sum_YW = np.zeros((self.num_treatments, self.num_outcomes))
            sum_WW = np.zeros((self.num_treatments, self.num_treatments))
            sum_weight = 0.0

            for sample in leaf_samples[i]:
                weight = data.get_weight(sample)
                outcome = data.get_outcomes(sample)
                treatment = data.get_treatments(sample)
                sum_Y += weight * outcome
                sum_W += weight * treatment
                sum_YW += weight * np
