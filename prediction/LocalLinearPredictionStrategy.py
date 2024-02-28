import numpy as np

class LocalLinearPredictionStrategy:
    def __init__(self, lambdas, weight_penalty, linear_correction_variables):
        """
        Initialize a LocalLinearPredictionStrategy.

        :param lambdas: A list of lambda values for ridge regression.
        :param weight_penalty: A boolean indicating whether to apply a weight penalty.
        :param linear_correction_variables: A list of indices for variables used in linear correction.
        """
        self.lambdas = lambdas
        self.weight_penalty = weight_penalty
        self.linear_correction_variables = linear_correction_variables

    def prediction_length(self):
        """
        Get the length of the prediction vector, which corresponds to the number of lambda values.

        :return: The number of lambda values.
        """
        return len(self.lambdas)

    def predict(self, sampleID, weights_by_sampleID, train_data, data):
        num_variables = len(self.linear_correction_variables)
        num_nonzero_weights = len(weights_by_sampleID)

        indices = list(weights_by_sampleID.keys())
        weights_vec = np.array(list(weights_by_sampleID.values()))

        X = np.ones((num_nonzero_weights, num_variables + 1))
        Y = np.zeros((num_nonzero_weights, 1))

        for i, index in enumerate(indices):
            for j, var in enumerate(self.linear_correction_variables):
                X[i, j + 1] = train_data.get(index, var) - data.get(sampleID, var)
            Y[i] = train_data.get_outcome(index)

        M_unpenalized = X.T @ np.diag(weights_vec) @ X

        predictions = []

        for lambda_val in self.lambdas:
            M = M_unpenalized.copy()
            if not self.weight_penalty:
                normalization = np.trace(M) / (num_variables + 1)
                M[1:, 1:] += lambda_val * normalization
            else:
                for j in range(1, num_variables + 1):
                    M[j, j] += lambda_val * M[j, j]

            local_coefficients = np.linalg.solve(M, X.T @ np.diag(weights_vec) @ Y)
            predictions.append(local_coefficients[0, 0])

        return predictions



    def compute_variance(self, sampleID, samples_by_tree, weights_by_sampleID, train_data, data, ci_group_size):
        lambda_val = self.lambdas[0]

        num_variables = len(self.linear_correction_variables)
        num_nonzero_weights = len(weights_by_sampleID)

        sample_index_map = {index: i for i, index in enumerate(weights_by_sampleID.keys())}
        indices = list(sample_index_map.keys())
        weights_vec = np.array(list(weights_by_sampleID.values()))

        X = np.ones((num_nonzero_weights, num_variables + 1))
        Y = np.zeros(num_nonzero_weights)

        for i, index in enumerate(indices):
            X[i, 0] = 1
            for j, var in enumerate(self.linear_correction_variables):
                X[i, j + 1] = train_data.get(index, var) - data.get(sampleID, var)
            Y[i] = train_data.get_outcome(index)

        M = X.T @ np.diag(weights_vec) @ X
        if not self.weight_penalty:
            normalization = np.trace(M) / (num_variables + 1)
            M[1:, 1:] += lambda_val * normalization
        else:
            for i in range(1, num_variables + 1):
                M[i, i] += lambda_val * M[i, i]

        theta = np.linalg.solve(M, X.T @ np.diag(weights_vec) @ Y)

        e_one = np.zeros(num_variables + 1)
        e_one[0] = 1.0
        zeta = np.linalg.solve(M, e_one)

        X_times_zeta = X @ zeta
        local_prediction = X @ theta
        pseudo_residual = X_times_zeta * (Y - local_prediction)

        num_good_groups = 0
        psi_squared = psi_grouped_squared = avg_score = 0

        for group in range(len(samples_by_tree) // ci_group_size):
            good_group = all(len(samples_by_tree[group * ci_group_size + j]) > 0 for j in range(ci_group_size))
            if not good_group:
                continue

            num_good_groups += 1
            group_psi = 0

            for j in range(ci_group_size):
                b = group * ci_group_size + j
                psi_1 = sum(pseudo_residual[sample_index_map[sample]] for sample in samples_by_tree[b]) / len(samples_by_tree[b])
                psi_squared += psi_1 * psi_1
                group_psi += psi_1

            group_psi /= ci_group_size
            psi_grouped_squared += group_psi * group_psi
            avg_score += group_psi

        avg_score /= num_good_groups
        var_between = psi_grouped_squared / num_good_groups - avg_score * avg_score
        var_total = psi_squared / (num_good_groups * ci_group_size) - avg_score * avg_score
        group_noise = (var_total - var_between) / (ci_group_size - 1)

        # Placeholder for bayes_debiaser.debias
        # var_debiased = bayes_debiaser.debias(var_between, group_noise, num_good_groups)
        var_debiased = var_between - group_noise  # Simplified version

        return [var_debiased]
