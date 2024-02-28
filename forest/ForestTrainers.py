class ForestTrainers:
    @staticmethod
    def instrumental_trainer(reduced_form_weight, stabilize_splits):
        relabeling_strategy = InstrumentalRelabelingStrategy(reduced_form_weight)
        splitting_rule_factory = InstrumentalSplittingRuleFactory() if stabilize_splits else RegressionSplittingRuleFactory()
        prediction_strategy = InstrumentalPredictionStrategy()
        return ForestTrainer(relabeling_strategy, splitting_rule_factory, prediction_strategy)

    @staticmethod
    def multi_causal_trainer(num_treatments, num_outcomes, stabilize_splits, gradient_weights):
        response_length = num_treatments * num_outcomes
        relabeling_strategy = MultiCausalRelabelingStrategy(response_length, gradient_weights)
        splitting_rule_factory = MultiCausalSplittingRuleFactory(response_length, num_treatments) if stabilize_splits else MultiRegressionSplittingRuleFactory(response_length)
        prediction_strategy = MultiCausalPredictionStrategy(num_treatments, num_outcomes)
        return ForestTrainer(relabeling_strategy, splitting_rule_factory, prediction_strategy)

    @staticmethod
    def quantile_trainer(quantiles):
        relabeling_strategy = QuantileRelabelingStrategy(quantiles)
        splitting_rule_factory = ProbabilitySplittingRuleFactory(len(quantiles) + 1)
        return ForestTrainer(relabeling_strategy, splitting_rule_factory, None)

    @staticmethod
    def probability_trainer(num_classes):
        relabeling_strategy = NoopRelabelingStrategy()
        splitting_rule_factory = ProbabilitySplittingRuleFactory(num_classes)
        prediction_strategy = ProbabilityPredictionStrategy(num_classes)
        return ForestTrainer(relabeling_strategy, splitting_rule_factory, prediction_strategy)

    @staticmethod
    def regression_trainer():
        relabeling_strategy = NoopRelabelingStrategy()
        splitting_rule_factory = RegressionSplittingRuleFactory()
        prediction_strategy = RegressionPredictionStrategy()
        return ForestTrainer(relabeling_strategy, splitting_rule_factory, prediction_strategy)

    @staticmethod
    def multi_regression_trainer(num_outcomes):
        relabeling_strategy = MultiNoopRelabelingStrategy(num_outcomes)
        splitting_rule_factory = MultiRegressionSplittingRuleFactory(num_outcomes)
        prediction_strategy = MultiRegressionPredictionStrategy(num_outcomes)
        return ForestTrainer(relabeling_strategy, splitting_rule_factory, prediction_strategy)

    @staticmethod
    def ll_regression_trainer(split_lambda, weight_penalty, overall_beta, ll_split_cutoff, ll_split_variables):
        relabeling_strategy = LLRegressionRelabelingStrategy(split_lambda, weight_penalty, overall_beta, ll_split_cutoff, ll_split_variables)
        splitting_rule_factory = RegressionSplittingRuleFactory()
        prediction_strategy = RegressionPredictionStrategy()
        return ForestTrainer(relabeling_strategy, splitting_rule_factory, prediction_strategy)

