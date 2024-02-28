import numpy as np


class Data:
    """
    A class to represent data with various indices for outcomes, treatments, instruments, etc.
    """

    def __init__(self, data, num_rows=None, num_cols=None):
        """
        Initialize the Data object with data and dimensions.

        :param data: Either a list of double values or a tuple containing a list of double values and a list of sizes.
        :param num_rows: Number of rows in the data.
        :param num_cols: Number of columns in the data.
        """
        if isinstance(data, tuple):
            self.data = data[0]
            self.num_rows = data[1][0]
            self.num_cols = data[1][1]
        else:
            if data is None:
                raise ValueError("Invalid data storage: None")
            self.data = data
            self.num_rows = num_rows
            self.num_cols = num_cols

        self.outcome_index = []
        self.treatment_index = []
        self.instrument_index = None
        self.weight_index = None
        self.causal_survival_numerator_index = None
        self.causal_survival_denominator_index = None
        self.censor_index = None
        self.disallowed_split_variables = set()

    def set_outcome_index(self, index):
        """
        Set the outcome index.

        :param index: A single index or a list of indices for the outcome.
        """
        if isinstance(index, int):
            index = [index]
        self.outcome_index = index
        self.disallowed_split_variables.update(index)

    def set_treatment_index(self, index):
        """
        Set the treatment index.

        :param index: A single index or a list of indices for the treatment.
        """
        if isinstance(index, int):
            index = [index]
        self.treatment_index = index
        self.disallowed_split_variables.update(index)

    def set_instrument_index(self, index):
        """
        Set the instrument index.

        :param index: The index for the instrument.
        """
        self.instrument_index = index
        self.disallowed_split_variables.add(index)

    def set_weight_index(self, index):
        """
        Set the weight index.

        :param index: The index for the weight.
        """
        self.weight_index = index
        self.disallowed_split_variables.add(index)

    def set_causal_survival_numerator_index(self, index):
        """
        Set the causal survival numerator index.

        :param index: The index for the causal survival numerator.
        """
        self.causal_survival_numerator_index = index
        self.disallowed_split_variables.add(index)

    def set_causal_survival_denominator_index(self, index):
        """
        Set the causal survival denominator index.

        :param index: The index for the causal survival denominator.
        """
        self.causal_survival_denominator_index = index
        self.disallowed_split_variables.add(index)

    def set_censor_index(self, index):
        """
        Set the censor index.

        :param index: The index for the censor.
        """
        self.censor_index = index
        self.disallowed_split_variables.add(index)

    def get_all_values(self, samples, var):
        """
        Retrieve and sort all values for a given variable from specified samples.

        :param samples: A list of sample indices.
        :param var: The variable index.
        :return: A tuple of sorted values and the corresponding sorted sample indices.
        """
        all_values = [self.get(sample, var) for sample in samples]

        # Argsort, handling NaNs (NaNs go to the front)
        index = sorted(range(len(all_values)), key=lambda i: (np.isnan(all_values[i]), all_values[i]))
        sorted_samples = [samples[i] for i in index]

        # Update all_values based on sorted samples
        all_values = [self.get(sample, var) for sample in sorted_samples]

        # Remove duplicates while handling NaNs
        unique_values = []
        for value in all_values:
            if not unique_values or not np.isclose(value, unique_values[-1], equal_nan=True):
                unique_values.append(value)

        return unique_values, sorted_samples

    def get_num_cols(self):
        """
        Get the number of columns in the data.

        :return: The number of columns.
        """
        return self.num_cols

    def get_num_rows(self):
        """
        Get the number of rows in the data.

        :return: The number of rows.
        """
        return self.num_rows

    def get_num_outcomes(self):
        """
        Get the number of outcomes.

        :return: The number of outcomes.
        """
        return len(self.outcome_index) if self.outcome_index else 1

    def get_num_treatments(self):
        """
        Get the number of treatments.

        :return: The number of treatments.
        """
        return len(self.treatment_index) if self.treatment_index else 1

    def get_disallowed_split_variables(self):
        """
        Get the set of disallowed split variables.

        :return: A set of disallowed split variables.
        """
        return self.disallowed_split_variables

    # Implementation of 'get' method is needed for this class to work properly.
