class TreeOptions:
    """
    A class to hold options for building and pruning decision trees.
    """

    def __init__(self, mtry, min_node_size, honesty, honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty):
        """
        Initialize TreeOptions.

        :param mtry: The number of variables to try at each split.
        :param min_node_size: The minimum size of a node.
        :param honesty: Whether to use honesty in tree building.
        :param honesty_fraction: The fraction of data to be used for honesty in tree building.
        :param honesty_prune_leaves: Whether to prune leaves based on honesty.
        :param alpha: The alpha parameter for tree building.
        :param imbalance_penalty: The imbalance penalty for tree building.
        """
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty

    def get_mtry(self):
        """Get the number of variables to try at each split."""
        return self.mtry

    def get_min_node_size(self):
        """Get the minimum size of a node."""
        return self.min_node_size

    def get_honesty(self):
        """Get whether honesty is used in tree building."""
        return self.honesty

    def get_honesty_fraction(self):
        """Get the fraction of data used for honesty in tree building."""
        return self.honesty_fraction

    def get_honesty_prune_leaves(self):
        """Get whether leaves are pruned based on honesty."""
        return self.honesty_prune_leaves

    def get_alpha(self):
        """Get the alpha parameter for tree building."""
        return self.alpha

    def get_imbalance_penalty(self):
        """Get the imbalance penalty for tree building."""
        return self.imbalance_penalty
