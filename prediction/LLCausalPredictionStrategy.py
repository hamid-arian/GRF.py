
import numpy as np
from scipy.linalg import solve_triangular

class LLCausalPredictionStrategy:
    def __init__(self, lambdas, weight_penalty, linear_correction_variables):
        self.lambdas = lambdas
        self.weight_penalty = weight_penalty
        self.linear_correction_variables = linear_correction_variables

    def prediction_length(self):
        return len(self.lambdas)

    def predict(self, sampleID, weights_by_sampleID, train_data, test_data):
        num_variables = len(self.linear_correction_variables)
        num_nonzero_weights = len(weights_by_sampleID)
        num_lambdas = len(self.lambdas)

        indices = list(weights_by_sampleID.keys())
        weights_vec = np.zeros(num_nonzero_weights)

        for i, index in enumerate(indices):
            weight = weights_by_sampleID[index]
            weights_vec[i] = weight

        dim_X = 2 * num_variables + 2
        X = np.zeros((num_nonzero_weights, dim_X))
        Y = np.zeros((num_nonzero_weights, 1))
        treatment_index = num_variables + 1

        for i in range(num_nonzero_weights):
            index = indices[i]
            treatment = train_data.get_treatment(index)

            X[i, 0] = 1

            for j in range(num_variables):
                current_predictor = self.linear_correction_variables[j]
                X[i, j + 1] = train_data.get(index, current_predictor) - test_data.get(sampleID, current_predictor)
                X[i, treatment_index + j + 1] = X[i, j + 1] * treatment

            X[i, treatment_index] = treatment
            Y[i] = train_data.get_outcome(index)

        M_unpenalized = np.dot(X.T, weights_vec * X)

        predictions = []
        for i in range(num_lambdas):
            lambda_val = self.lambdas[i]
            M = M_unpenalized.copy()

            if not self.weight_penalty:
                normalization = np.trace(M_unpenalized) / dim_X

                for j in range(1, dim_X):
                    if j != treatment_index:
                        M[j, j] += lambda_val * normalization
            else:
                for j in range(1, dim_X):
                    if j != treatment_index:
                        M[j, j] += lambda_val * M[j, j]

            local_coefficients = solve_triangular(M, np.dot(X.T, weights_vec * Y))

            predictions.append(local_coefficients[treatment_index])

        return predictions

    def compute_variance(self, sampleID, samples_by_tree, weights_by_sampleID, train_data, test_data, ci_group_size):
        lambda_val = self.lambdas[0]
        num_variables = len(self.linear_correction_variables)
        num_nonzero_weights = len(weights_by_sampleID)

        sample_index_map = [0] * train_data.get_num_rows()
        indices = list(weights_by_sampleID.keys())
        weights_vec = np.zeros(num_nonzero_weights)

        for i, index in enumerate(indices):
            sample_index_map[index] = i
            weights_vec[i] = weights_by_sampleID[index]

        dim_X = 2 * num_variables + 2
        X = np.zeros((num_nonzero_weights, dim_X))
        Y = np.zeros((num_nonzero_weights, 1))
        treatment_index = num_variables + 1

        for i in range(num_nonzero_weights):
            index = indices[i]
            treatment = train_data.get_treatment(index)

            X[i, 0] = 1

            for j in range(num_variables):
                current_predictor = self.linear_correction_variables[j]
                X[i, j + 1] = train_data.get(index, current_predictor) - test_data.get(sampleID, current_predictor)
                X[i, treatment_index + j + 1] = X[i, j + 1] * treatment

            X[i, treatment_index] = treatment
            Y[i] = train_data.get_outcome(index)

        M_unpenalized = np.dot(X.T, weights_vec * X)
        M = M_unpenalized.copy()

        if not self.weight_penalty:
            normalization = np.trace(M_unpenalized) / dim_X

            for j in range(1, dim_X):
                if j != treatment_index:
                    M[j, j] += lambda_val * normalization
        else:
            for j in range(1, dim_X):
                if j != treatment_index:
                    M[j, j] += lambda_val * M[j, j]

        theta = solve_triangular(M, np.dot(X.T, weights_vec * Y))

        e_trt = np.zeros(dim_X)
        e_trt[treatment_index] = 1.0
        zeta = solve_triangular(M, e_trt)

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

class Data:
    def __init__(self):
        pass

class BayesDebiaser:
    def debias(self, var_between, group_noise, num_good_groups):
        # Implement your debiasing logic here
        pass

# Usage example
if __name__ == "__main__":
    # Create instances of classes
    lambdas = [0.1, 0.5, 1
