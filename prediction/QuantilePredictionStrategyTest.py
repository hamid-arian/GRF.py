import unittest
import numpy as np

class TestQuantilePredictionStrategy(unittest.TestCase) :

    def test_simple_quantile_prediction(self) :
    weights_by_sample = {
        0: 0.0, 1 : 0.1, 2 : 0.2, 3 : 0.1, 4 : 0.1,
        5 : 0.1, 6 : 0.2, 7 : 0.1, 8 : 0.0, 9 : 0.1
}
outcomes = [-9.99984, -7.36924, 5.11211, -0.826997, 0.655345,
-5.62082, -9.05911, 3.57729, 3.58593, 8.69386]

data = Data(outcomes, 10, 1)
data.set_outcome_index(0)

prediction_strategy = QuantilePredictionStrategy([0.25, 0.5, 0.75])
predictions = prediction_strategy.predict(0, weights_by_sample, data, data)

expected_predictions = [-7.36924, -0.826997, 5.11211]
self.assertEqual(predictions, expected_predictions)

def test_prediction_with_skewed_quantiles(self) :
    weights_by_sample = {
        0: 0.0, 1 : 0.1, 2 : 0.2, 3 : 0.1, 4 : 0.1,
        5 : 0.1, 6 : 0.2, 7 : 0.1, 8 : 0.0, 9 : 0.1
}
outcomes = [-1.99984, -0.36924, 0.11211, -1.826997, 1.655345,
-1.62082, -0.05911, 0.57729, 0.58593, 1.69386]

data = Data(outcomes, 10, 1)
data.set_outcome_index(0)

prediction_strategy = QuantilePredictionStrategy([0.5, 0.75, 0.80, 0.90])
predictions = prediction_strategy.predict(42, weights_by_sample, data, data)

# Check that all predictions fall within a reasonable range.
for prediction in predictions :
self.assertTrue(-2.0 < prediction < 2.0)

    def test_prediction_with_repeated_quantiles(self) :
    weights_by_sample = {
        0: 0.0, 1 : 0.1, 2 : 0.2, 3 : 0.1, 4 : 0.1,
        5 : 0.1, 6 : 0.2, 7 : 0.1, 8 : 0.0, 9 : 0.1
}
outcomes = [-9.99984, -7.36924, 5.11211, -0.826997, 0.655345,
-5.62082, -9.05911, 3.57729, 3.58593, 8.69386]

data = Data(outcomes, 10, 1)
data.set_outcome_index(0)

first_predictions = QuantilePredictionStrategy([0.5]).predict(42, weights_by_sample, data, data)
second_predictions = QuantilePredictionStrategy([0.25, 0.5, 0.75]).predict(42, weights_by_sample, data, data)
third_predictions = QuantilePredictionStrategy([0.5, 0.5, 0.5]).predict(42, weights_by_sample, data, data)

self.assertEqual(first_predictions[0], second_predictions[1])
for prediction in third_predictions :
self.assertEqual(prediction, first_predictions[0])

if __name__ == '__main__' :
    unittest.main()
