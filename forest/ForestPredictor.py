class ForestPredictor:
    def __init__(self, num_threads, strategy):
        """
        Initialize a ForestPredictor.

        :param num_threads: The number of threads to use for prediction.
        :param strategy: The prediction strategy (either Default or Optimized).
        """
        self.tree_traverser = TreeTraverser(num_threads)
        if isinstance(strategy, DefaultPredictionStrategy):
            self.prediction_collector = DefaultPredictionCollector(strategy, num_threads)
        elif isinstance(strategy, OptimizedPredictionStrategy):
            self.prediction_collector = OptimizedPredictionCollector(strategy, num_threads)
        else:
            raise ValueError("Unknown prediction strategy type.")

    def predict(self, forest, train_data, data, estimate_variance=False, oob_prediction=False):
        """
        Make predictions using the forest.

        :param forest: The Forest object.
        :param train_data: The training data used to train the forest.
        :param data: The data to predict.
        :param estimate_variance: Whether to estimate variance in predictions.
        :param oob_prediction: Whether to make out-of-bag predictions.
        :return: A list of predictions.
        """
        if estimate_variance and forest.get_ci_group_size() <= 1:
            raise RuntimeError("To estimate variance during prediction, the forest must be trained with ci_group_size greater than 1.")

        leaf_nodes_by_tree = self.tree_traverser.get_leaf_nodes(forest, data, oob_prediction)
        trees_by_sample = self.tree_traverser.get_valid_trees_by_sample(forest, data, oob_prediction)

        return self.prediction_collector.collect_predictions(forest, train_data, data, leaf_nodes_by_tree, trees_by_sample, estimate_variance, oob_prediction)

    def predict_oob(self, forest, data, estimate_variance=False):
        """
        Make out-of-bag predictions using the forest.

        :param forest: The Forest object.
        :param data: The data to predict.
        :param estimate_variance: Whether to estimate variance in predictions.
        :return: A list of predictions.
        """
        return self.predict(forest, data, data, estimate_variance, True)
