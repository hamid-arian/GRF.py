
class OptimizedPredictionStrategy:
    def prediction_length(self):
        """
        The number of values in a prediction, e.g., 1 for regression,
        or the number of quantiles for quantile forests.
        """
        raise NotImplementedError

    def predict(self, average_prediction_values):
        """
        Computes a prediction for a single test sample.

        average_prediction_values: the 'prediction values' computed during
        training, averaged across all leaves this test sample landed in.
        """
        raise NotImplementedError

    def compute_variance(self, average_prediction_values, leaf_prediction_values, ci_group_size):
        """
        Computes a prediction variance estimate for a single test sample.

        average_prediction_values: the 'prediction values' computed during training,
        averaged across all leaves this test sample landed in.
        leaf_prediction_values: the individual 'prediction values' for each leaf this test
        sample landed in. There will be one entry per tree, even if that tree was OOB or
        the leaf was empty.
        ci_group_size: the size of the tree groups used to train the forest. This
        parameter is used when computing within vs. across group variance.
        """
        raise NotImplementedError

    def prediction_value_length(self):
        """
        The number of types of precomputed prediction values. For regression
        this is 1 (the average outcome), whereas for instrumental forests this
        is larger, as it includes the average treatment, average instrument, etc.
        """
        raise NotImplementedError

    def precompute_prediction_values(self, leaf_samples, data):
        """
        This method is called during training on each tree to precompute
        summary values to be used during prediction.

        As an example, the regression prediction strategy computes the average outcome in
        each leaf so that it does not need to recompute these values during every prediction.
        """
        raise NotImplementedError

    def compute_error(self, sample, average, leaf_values, data):
        """
        Computes a pair of estimates for (out-of-bag debiased error, monte-carlo error) for a single sample.
        The 'debiased error' is the expected error for a forest containing an infinite number of trees.
        The 'monte-carlo error' is the error inherent in the algorithm randomization, i.e., a measure of how
        different predictions from two forests grown on the same data can be.

        sample: index of the observation
        leaf_values: collected prediction values from all leaves across forests
        observations: depending on the forest type, this may contain output, treatment
        and/or instrument values. These are used to compute an estimate of the
        error given leaf_values.
        """
        raise NotImplementedError

# Placeholder class for Data
class Data:
    pass
