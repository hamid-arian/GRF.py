import numpy as np
from numpy.linalg import solve, LinAlgError

class LocalLinearPredictionStrategy:
    def __init__(self, lambdas, weight_penalty, linear_correction_variables):
        self.lambdas = lambdas
        self.weight_penalty = weight_penalty
        self.linear_correction_variables = linear_correction_variables

    def prediction_length(self):
        return len(self.lambdas)

    def predict(self, sampleID, weights_by_sampleID, train_data, data):
        num_variables = len(self.linear_correction_variables)
        num_nonzero_weights = len(weights_by_sampleID)

        indices = list(weights_by_sampleID.keys())
        weights_vec = np.zeros(num_nonzero_weights)

        for i, index in enumerate(indices):
            weights_vec[i] = weights_by_sampleID[index]

        X = np.zeros((num_nonzero_weights, num_variables + 1))
        Y = np.zeros((num_nonzero_weights, 1))

        for i in range(num_nonzero_weights):
            for j in range(num_variables):
                current_predictor = self.linear_correction_variables[j]
                X[i, j + 1] = train_data.get(indices[i], current_predictor) - data.get(sampleID, current_predictor)

            Y[i] = train_data.get_outcome(indices[i])
            X[i, 0] = 1

        M_unpenalized = np.dot(np.dot(X.transpose(), np.diag(weights_vec)), X)

        num_lambdas = len(self.lambdas)
        predictions = []

        for i in range(num_lambdas):
            lambda_val = self.lambdas[i]
            M = M_unpenalized.copy()

            if not self.weight_penalty:
                normalization = np.trace(M) / (num_variables + 1)

                for j in range(1, num_variables + 1):
                    M[j, j] += lambda_val * normalization
            else:
                for j in range(1, num_variables + 1):
                    M[j, j] += lambda_val * M[j, j]

            try:
                local_coefficients = solve(M, np.dot(np.dot(X.transpose(), np.diag(weights_vec)), Y))
                predictions.append(local_coefficients[0])
            except LinAlgError:
                # Handle singular matrix (possible in some cases)
                predictions.append(0.0)

        return predictions

    def compute_variance(self, sampleID, samples_by_tree, weights_by_sampleID, train_data, data, ci_group_size):
        lambda_val = self.lambdas[0]
        num_variables = len(self.linear_correction_variables)
        num_nonzero_weights = len(weights_by_sampleID)

        sample_index_map = {index: i for i, index in enumerate(weights_by_sampleID.keys())}
        indices = list(weights_by_sampleID.keys())
        weights_vec = np.zeros(num_nonzero_weights)

        for i, index in enumerate(indices):
            weights_vec[i] = weights_by_sampleID[index]

        X = np.zeros((num_nonzero_weights, num_variables + 1))
        Y = np.zeros((num_nonzero_weights, 1))

        for i in range(num_nonzero_weights):
            for j in range(num_variables):
                current_predictor = self.linear_correction_variables[j]
                X[i, j + 1] = train_data.get(indices[i], current_predictor) - data.get(sampleID, current_predictor)

            Y[i] = train_data.get_outcome(indices[i])
            X[i, 0] = 1

        M = np.dot(np.dot(X.transpose(), np.diag(weights_vec)), X)

        if not self.weight_penalty:
            normalization = np.trace(M) / (num_variables + 1)

            for i in range(1, num_variables + 1):
                M[i, i] += lambda_val * normalization
        else:
            for i in range(1, num_variables + 1):
                M[i, i] += lambda_val * M[i, i]

        try:
            theta = solve(M, np.dot(np.dot(X.transpose(), np.diag(weights_vec)), Y))
        except LinAlgError:
            # Handle singular matrix (possible in some cases)
            return [0.0]

        e_one = np.zeros(num_variables + 1)
        e_one[0] = 1.0
        zeta = solve(M, e_one)

        X_times_zeta = np.dot(X, zeta)
        local_prediction = np.dot(X, theta)
        pseudo_residual = np.zeros(num_nonzero_weights)

        for i in range(num_nonzero_weights):
            pseudo_residual[i] = X_times_zeta[i] * (Y[i] - local_prediction[i])

        num_good_groups = 0
        psi_squared = 0
        psi_grouped_squared = 0
        avg_score = 0

        for group in range(len(samples_by_tree) // ci_group_size):
            good_group = True

            for j in range(ci_group_size):
                if len(samples_by_tree[group * ci_group_size + j]) == 0:
                    good_group = False

            if not good_group:
                continue

            num_good_groups += 1
            group_psi = 0

            for j in range(ci_group_size):
                b = group * ci_group_size + j
                psi_1 = 0

                for sample in samples_by_tree[b]:
                    psi_1 += pseudo_residual[sample_index_map[sample]]

                psi_1 /= len(samples_by_tree[b])
                psi_squared += psi_1 * psi_1
                group_psi += psi_1

            group_psi /= ci_group_size
            psi_grouped_squared += group_psi * group_psi
            avg_score += group_psi

        avg_score /= num_good_groups
        var_between = psi_grouped_squared / num_good_groups - avg_score * avg_score
        var_total = psi_squared / (num_good_groups * ci_group_size) - avg_score * avg_score
        group_noise = (var_total - var_between) / (ci_group_size - 1)
        var_debiased = var_between - group_noise

        return [var_debiased]
