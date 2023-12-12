# forest_predictor.py

from prediction.collector.default_prediction_collector import DefaultPredictionCollector
from prediction.collector.optimized_prediction_collector import OptimizedPredictionCollector
from prediction.strategy.default_prediction_strategy import DefaultPredictionStrategy
from prediction.strategy.optimized_prediction_strategy import OptimizedPredictionStrategy
from prediction.prediction import Prediction
from prediction.collector.tree_traverser import TreeTraverser
from forest.forest import Forest
from commons.data import Data
import threading

class ForestPredictor:
    def __init__(self, num_threads, strategy):
        self.tree_traverser = TreeTraverser(num_threads)
        self.prediction_collector = None

        if isinstance(strategy, DefaultPredictionStrategy):
            self.prediction_collector = DefaultPredictionCollector(strategy, num_threads)
        elif isinstance(strategy, OptimizedPredictionStrategy):
            self.prediction_collector = OptimizedPredictionCollector(strategy, num_threads)

    def predict(self, forest, train_data, data, estimate_variance):
        return self._predict(forest, train_data, data, estimate_variance, False)

    def predict_oob(self, forest, data, estimate_variance):
        return self._predict(forest, data, data, estimate_variance, True)

    def _predict(self, forest, train_data, data, estimate_variance, oob_prediction):
        if estimate_variance and forest.get_ci_group_size() <= 1:
            raise RuntimeError("To estimate variance during prediction, the forest must"
                               " be trained with ci_group_size greater than 1.")

        leaf_nodes_by_tree = self.tree_traverser.get_leaf_nodes(forest, data, oob_prediction)
        trees_by_sample = self.tree_traverser.get_valid_trees_by_sample(forest, data, oob_prediction)

        return self.prediction_collector.collect_predictions(
            forest, train_data, data,
            leaf_nodes_by_tree, trees_by_sample,
            estimate_variance, oob_prediction
        )
