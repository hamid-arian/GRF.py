import unittest

class TestInstrumentalPredictionStrategy(unittest.TestCase):

    def test_flipping_signs_of_treatment(self):
        averages = [-1.1251472, 0.3, 0.5, -0.1065444, 0.2, 1, 1]
        flipped_averages = [-1.1251472, 0.7, 0.5, -0.1065444, 0.3, 1, 1]

        prediction_strategy = InstrumentalPredictionStrategy()
        first_prediction = prediction_strategy.predict(averages)
        second_prediction = prediction_strategy.predict(flipped_averages)

        self.assertEqual(len(first_prediction), 1)
        self.assertEqual(len(second_prediction), 1)
        self.assertAlmostEqual(first_prediction[0], -second_prediction[0], delta=1.0e-10)

    def test_instrumental_variance_estimates_are_positive(self):
        averages = [1, 0, 4.5, 2, 0.75, 1, 1]
        leaf_values = [
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 1, 1],
            [-2, -3, 5, -3, -1, 1, 1],
            [1, 0, 1, 2, 1, 1, 1]
        ]

        prediction_strategy = InstrumentalPredictionStrategy()
        variance = prediction_strategy.compute_variance(
            averages, PredictionValues(leaf_values, 5), 2
        )

        self.assertEqual(len(variance), 1)
        self.assertTrue(variance[0] > 0)

    def test_scaling_outcome_scales_instrumental_variance(self):
        averages = [1, 0, 4.5, 2, 0.75, 1, 1]
        leaf_values = [
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 1, 1],
            [-2, -3, 5, -3, -1, 1, 1],
            [1, 0, 1, 2, 1, 1, 1]
        ]

        scaled_average = [2, 0, 4.5, 4, 0.75, 1, 1]
        scaled_leaf_values = [
            [2, 1, 1, 2, 1, 1, 1],
            [4, 2, 2, 4, 2, 1, 1],
            [-4, -3, 5, -6, -1, 1, 1],
            [2, 0, 1, 4, 1, 1, 1]
        ]

        prediction_strategy = InstrumentalPredictionStrategy()
        first_variance = prediction_strategy.compute_variance(
            averages, PredictionValues(leaf_values, 5), 2
        )
        second_variance = prediction_strategy.compute_variance(
            scaled_average, PredictionValues(scaled_leaf_values, 5), 2
        )

        self.assertEqual(len(first_variance), 1)
        self.assertEqual(len(second_variance), 1)
        self.assertAlmostEqual(first_variance[0], second_variance[0] / 4, delta=1.0e-10)

    def test_monte_carlo_errors_are_nonzero(self):
        average = [2.725, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        leaf_values = [
            [2, 1, 1, 2, 1, 2, 1],
            [4, 2, 2, 4, 2, 1, 1],
            [-4, -3, 5, -6, -1, 0, 1],
            [2, 0, 1, 4, 1, 0, 1],
            [2, 0, 1, 4, 1, 3, 1],
            [2, 0, 1, 3, 4, 1, 1]
        ]

        outcomes = [
            6.4, 1.0, 1.4, 1.0, 0.0, 1.6,
            1.4, 2.0, 2.4, 2.0, 1.0, 5.5,
            2.4, 3.0, 3.4, 3.0, 2.0, 4.4,
            3.4, 2.0, 3.4, 4.0, 3.0, 3.3,
            4.4, 3.0, 14.4, 5.0, 4.0, 2.2,
            3.4, 9.0, 16.4, 6.0, 5.0, 1.1
        ]

        data = Data(outcomes, 6, 6)
        data.set_outcome_index(0)
        data.set_instrument_index(1)
        data.set_treatment_index(1)

        prediction_strategy = InstrumentalPredictionStrategy()
        sample = 0

        errors = prediction_strategy.compute_error(
            sample, average, PredictionValues(leaf_values, 3), data
        )

        mc_error = errors[0].second

        self.assertGreater(mc_error, 0)

if __name__ == '__main__':
    unittest.main()
