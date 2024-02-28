import numpy as np

class LLRegressionRelabelingStrategy:
    def __init__(self, split_lambda, weight_penalty, overall_beta, ll_split_cutoff, ll_split_variables):
        """
        Initialize the LLRegressionRelabelingStrategy.

        :param split_lambda: The ridge penalty lambda.
        :param weight_penalty: Flag to indicate if a weight penalty is used.
        :param overall_beta: Coefficients for the global model.
        :param ll_split_cutoff: The cutoff for switching between local and global models.
        :param ll_split_variables: Variables considered for the split.
        """
        self.split_lambda = split_lambda
        self.weight_penalty = weight_penalty
        self.overall_beta = np.array(overall_beta)
        self.ll_split_cutoff = ll_split_cutoff
        self.ll_split_variables = ll_split_variables

    def relabel(self, samples, data, responses_by_sample):
        num_variables = len(self.ll_split_variables)
        num_data_points = len(samples)

        X = np.ones((num_data_points, num_variables + 1))
        Y = np.array([data.get_outcome(sample) for sample in samples])

        for i, sample in enumerate(samples):
            for j, var in enumerate(self.ll_split_variables):
                X[i, j + 1] = data.get(sample, var)

        if num_data_points < self.ll_split_cutoff:
            leaf_predictions = X @ self.overall_beta
        else:
            M = X.T @ X
            if self.weight_penalty:
                for j in range(1, num_variables + 1):
                    M[j, j] += self.split_lambda * M[j, j]
            else:
                normalization = np.trace(M) / (num_variables + 1)
                M[1:, 1:] += self.split_lambda * normalization

            local_coefficients = np.linalg.solve(M, X.T @ Y)
            leaf_predictions = X @ local_coefficients

        for i, sample in enumerate(samples):
            prediction = leaf_predictions[i]
            residual = prediction - data.get_outcome(sample)
            responses_by_sample[sample, 0] = residual

        return False
