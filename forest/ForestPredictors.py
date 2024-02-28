class ForestPredictors:
    @staticmethod
    def instrumental_predictor(num_threads):
        num_threads = ForestOptions.validate_num_threads(num_threads)
        prediction_strategy = InstrumentalPredictionStrategy()
        return ForestPredictor(num_threads, prediction_strategy)

    @staticmethod
    def multi_causal_predictor(num_threads, num_treatments, num_outcomes):
        num_threads = ForestOptions.validate_num_threads(num_threads)
        prediction_strategy = MultiCausalPredictionStrategy(num_treatments, num_outcomes)
        return ForestPredictor(num_threads, prediction_strategy)

    @staticmethod
    def quantile_predictor(num_threads, quantiles):
        num_threads = ForestOptions.validate_num_threads(num_threads)
        prediction_strategy = QuantilePredictionStrategy(quantiles)
        return ForestPredictor(num_threads, prediction_strategy)

    @staticmethod
    def probability_predictor(num_threads, num_classes):
        num_threads = ForestOptions.validate_num_threads(num_threads)
        prediction_strategy = ProbabilityPredictionStrategy(num_classes)
        return ForestPredictor(num_threads, prediction_strategy)

    @staticmethod
    def regression_predictor(num_threads):
        num_threads = ForestOptions.validate_num_threads(num_threads)
        prediction_strategy = RegressionPredictionStrategy()
        return ForestPredictor(num_threads, prediction_strategy)

    @staticmethod
    def multi_regression_predictor(num_threads, num_outcomes):
        num_threads = ForestOptions.validate_num_threads(num_threads)
        prediction_strategy = MultiRegressionPredictionStrategy(num_outcomes)
        return ForestPredictor(num_threads, prediction_strategy)

    @staticmethod
    def ll_regression_predictor(num_threads, lambdas, weight_penalty, linear_correction_variables):
        num_threads = ForestOptions.validate_num_threads(num_threads)
        prediction_strategy = LocalLinearPredictionStrategy(lambdas, weight_penalty, linear_correction_variables)
        return ForestPredictor(num_threads, prediction_strategy)

    @staticmethod
    def ll_causal_predictor(num_threads, lambdas, weight_penalty, linear_correction_variables):
        num_threads = ForestOptions.validate_num_threads(num_threads)
        prediction_strategy = LLCausalPredictionStrategy(lambdas, weight_penalty, linear_correction_variables)
        return ForestPredictor(num_threads, prediction_strategy)

    @staticmethod
    def survival_predictor(num_threads, num_failures, prediction_type):
        num_threads = ForestOptions.validate_num_threads(num_threads)
        prediction_strategy = SurvivalPredictionStrategy(num_failures, prediction_type)
        return ForestPredictor(num_threads, prediction_strategy)

    @staticmethod
    def causal_survival_predictor(num_threads):
        num_threads = ForestOptions.validate_num_threads(num_threads)
        prediction_strategy = CausalSurvivalPredictionStrategy()
        return ForestPredictor(num_threads, prediction_strategy)

# Placeholder for prediction strategies and ForestPredictor
# class InstrumentalPredictionStrategy: ...
# class MultiCausalPredictionStrategy: ...
# ... other prediction strategies ...
# class ForestPredictor: ...
