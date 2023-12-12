# forest_predictors.py

from forest.forest_predictor import ForestPredictor
from prediction.instrumental_prediction_strategy import InstrumentalPredictionStrategy
from prediction.multi_causal_prediction_strategy import MultiCausalPredictionStrategy
from prediction.quantile_prediction_strategy import QuantilePredictionStrategy
from prediction.probability_prediction_strategy import ProbabilityPredictionStrategy
from prediction.regression_prediction_strategy import RegressionPredictionStrategy
from prediction.multi_regression_prediction_strategy import MultiRegressionPredictionStrategy
from prediction.local_linear_prediction_strategy import LocalLinearPredictionStrategy
from prediction.ll_causal_prediction_strategy import LLCausalPredictionStrategy
from prediction.survival_prediction_strategy import SurvivalPredictionStrategy
from prediction.causal_survival_prediction_strategy import CausalSurvivalPredictionStrategy

def instrumental_predictor(num_threads):
    num_threads = ForestPredictor.validate_num_threads(num_threads)
    prediction_strategy = InstrumentalPredictionStrategy()
    return ForestPredictor(num_threads, prediction_strategy)

def multi_causal_predictor(num_threads, num_treatments, num_outcomes):
    num_threads = ForestPredictor.validate_num_threads(num_threads)
    prediction_strategy = MultiCausalPredictionStrategy(num_treatments, num_outcomes)
    return ForestPredictor(num_threads, prediction_strategy)

def quantile_predictor(num_threads, quantiles):
    num_threads = ForestPredictor.validate_num_threads(num_threads)
    prediction_strategy = QuantilePredictionStrategy(quantiles)
    return ForestPredictor(num_threads, prediction_strategy)

def probability_predictor(num_threads, num_classes):
    num_threads = ForestPredictor.validate_num_threads(num_threads)
    prediction_strategy = ProbabilityPredictionStrategy(num_classes)
    return ForestPredictor(num_threads, prediction_strategy)

def regression_predictor(num_threads):
    num_threads = ForestPredictor.validate_num_threads(num_threads)
    prediction_strategy = RegressionPredictionStrategy()
    return ForestPredictor(num_threads, prediction_strategy)

def multi_regression_predictor(num_threads, num_outcomes):
    num_threads = ForestPredictor.validate_num_threads(num_threads)
    prediction_strategy = MultiRegressionPredictionStrategy(num_outcomes)
    return ForestPredictor(num_threads, prediction_strategy)

def ll_regression_predictor(num_threads, lambdas, weight_penalty, linear_correction_variables):
    num_threads = ForestPredictor.validate_num_threads(num_threads)
    prediction_strategy = LocalLinearPredictionStrategy(lambdas, weight_penalty, linear_correction_variables)
    return ForestPredictor(num_threads, prediction_strategy)

def ll_causal_predictor(num_threads, lambdas, weight_penalty, linear_correction_variables):
    num_threads = ForestPredictor.validate_num_threads(num_threads)
    prediction_strategy = LLCausalPredictionStrategy(lambdas, weight_penalty, linear_correction_variables)
    return ForestPredictor(num_threads, prediction_strategy)

def survival_predictor(num_threads, num_failures, prediction_type):
    num_threads = ForestPredictor.validate_num_threads(num_threads)
    prediction_strategy = SurvivalPredictionStrategy(num_failures, prediction_type)
    return ForestPredictor(num_threads, prediction_strategy)

def causal_survival_predictor(num_threads):
    num_threads = ForestPredictor.validate_num_threads(num_threads)
    prediction_strategy = CausalSurvivalPredictionStrategy()
    return ForestPredictor(num_threads, prediction_strategy)
