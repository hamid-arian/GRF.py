import numpy as np


class MultiCausalRelabelingStrategy:
    def __init__(self, response_length, gradient_weights):
        self.response_length = response_length
        if not gradient_weights:
            self.gradient_weights = [1.0] * response_length
        elif len(gradient_weights) != response_length:
            raise RuntimeError("Optional gradient weights vector must be the same length as response_length.")
        else:
            self.gradient_weights = gradient_weights

    def relabel(self, samples, data, responses_by_sample):
        # Prepare the relevant averages.
        num_samples = len(samples)
        num_treatments = data.get_num_treatments()
        num_outcomes = data.get_num_outcomes()
        if num_samples <= num_treatments:
            return True

        Y_centered = np.zeros((num_samples, num_outcomes))
        W_centered = np.zeros((num_samples, num_treatments))
        weights = np.zeros(num_samples)
        Y_mean = np.zeros(num_outcomes)
        W_mean = np.zeros(num_treatments)
        sum_weight = 0

        for i, sample in enumerate(samples):
            weight = data.get_weight(sample)
            outcome = data.get_outcomes(sample)
            treatment = data.get_treatments(sample)

            Y_centered[i, :] = outcome
            W_centered[i, :] = treatment
            weights[i] = weight

            Y_mean += weight * outcome
            W_mean += weight * treatment
            sum_weight += weight

        Y_mean /= sum_weight
        W_mean /= sum_weight
        Y_centered -= Y_mean
        W_centered -= W_mean

        if np.abs(sum_weight) <= 1e-16:
            return True

        WW_bar = np.dot(W_centered.T, np.dot(np.diag(weights), W_centered))

        # Calculate the treatment effect.
        if np.isclose(np.linalg.det(WW_bar), 0.0, atol=1.0e-10):
            return True

        A_p_inv = np.linalg.inv(WW_bar)
        beta = np.dot(A_p_inv, np.dot(W_centered.T, np.dot(np.diag(weights), Y_centered)))

        rho_weight = np.dot(W_centered, A_p_inv.T)
        residual = Y_centered - np.dot(W_centered, beta)

        # Create the new outcomes, eq (20) in https://arxiv.org/pdf/1610.01271.pdf
        j = 0
        for i, sample in enumerate(samples):
            for outcome in range(num_outcomes):
                for treatment in range(num_treatments):
                    responses_by_sample[sample, j] = (
                            rho_weight[i, treatment] * residual[i, outcome] * self.gradient_weights[j]
                    )
                    j += 1

        return False

    def get_response_length(self):
        return self.response_length
