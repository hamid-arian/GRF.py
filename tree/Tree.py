class Tree:
    """
    A class representing a decision tree.
    """

    def __init__(self, root_node, child_nodes, leaf_samples, split_vars, split_values, drawn_samples, send_missing_left, prediction_values):
        """
        Initialize a Tree object.

        :param root_node: The index of the root node.
        :param child_nodes: A list of child nodes for each node.
        :param leaf_samples: A list of samples for each leaf node.
        :param split_vars: The variables used for splitting at each node.
        :param split_values: The values used for splitting at each node.
        :param drawn_samples: The samples drawn for the tree.
        :param send_missing_left: Flags indicating whether missing values go left at each node.
        :param prediction_values: The prediction values for each node.
        """
        self.root_node = root_node
        self.child_nodes = child_nodes
        self.leaf_samples = leaf_samples
        self.split_vars = split_vars
        self.split_values = split_values
        self.drawn_samples = drawn_samples
        self.send_missing_left = send_missing_left
        self.prediction_values = prediction_values

    def get_root_node(self):
        """Get the root node of the tree."""
        return self.root_node

    def get_child_nodes(self):
        """Get the child nodes of the tree."""
        return self.child_nodes

    def get_leaf_samples(self):
        """Get the leaf samples of the tree."""
        return self.leaf_samples

    def get_split_vars(self):
        """Get the split variables of the tree."""
        return self.split_vars

    def get_split_values(self):
        """Get the split values of the tree."""
        return self.split_values

    def get_drawn_samples(self):
        """Get the drawn samples of the tree."""
        return self.drawn_samples

    def get_send_missing_left(self):
        """Get the flags for sending missing values left."""
        return self.send_missing_left

    def get_prediction_values(self):
        """Get the prediction values of the tree."""
        return self.prediction_values

    def find_leaf_nodes(self, data, samples):
        """
        Find the leaf nodes for the given samples.

        :param data: The data used for the tree.
        :param samples: The samples to find leaf nodes for.
        :return: A list of leaf nodes for each sample.
        """
        prediction_leaf_nodes = [0] * data.get_num_rows()

        for sample in samples:
            node = self.find_leaf_node(data, sample)
            prediction_leaf_nodes[sample] = node

        return prediction_leaf_nodes

    def set_leaf_samples(self, leaf_samples):
        """Set the leaf samples of the tree."""
        self.leaf_samples = leaf_samples

    def set_prediction_values(self, prediction_values):
        """Set the prediction values of the tree."""
        self.prediction_values = prediction_values

    def find_leaf_node(self, data, sample):
        """
        Find the leaf node for a given sample.

        :param data: The data used for the tree.
        :param sample: The sample to find the leaf node for.
        :return: The leaf node index.
        """
        node = self.root_node
        while True:
            if self.is_leaf(node):
                break

            split_var = self.split_vars[node]
            split_val = self.split_values[node]
            value = data.get(sample, split_var)
            send_na_left = self.send_missing_left[node]

            if value <= split_val or (send_na_left and math.isnan(value)) or (math.isnan(split_val) and math.isnan(value)):
                node = self.child_nodes[0][node]
            else:
                node = self.child_nodes[1][node]

        return node

    def honesty_prune_leaves(self):
        """
        Prune the leaves of the tree based on honesty criteria.
        """
        num_nodes = len(self.leaf_samples)
        for n in range(num_nodes, self.root_node, -1):
            node = n - 1
            if self.is_leaf(node):
                continue

            left_child = self.child_nodes[0][node]
            if not self.is_leaf(left_child):
                self.prune_node(left_child)

            right_child = self.child_nodes[1][node]
            if not self.is_leaf(right_child):
                self.prune_node(right_child)

        self.prune_node(self.root_node)

    def prune_node(self, node):
        """
        Prune a specific node.

        :param node: The node to prune.
        """
        left_child = self.child_nodes[0][node]
        right_child = self.child_nodes[1][node]

        if self.is_empty_leaf(left_child) or self.is_empty_leaf(right_child):
            self.child_nodes[0][node] = 0
            self.child_nodes[1][node] = 0

            if not self.is_empty_leaf(left_child):
                node = left_child
            elif not self.is_empty_leaf(right_child):
                node = right_child

    def is_leaf(self, node):
        """
        Check if a node is a leaf.

        :param node: The node to check.
        :return: True if the node is a leaf, False otherwise.
        """
        return self.child_nodes[0][node] == 0 and self.child_nodes[1][node] == 0

    def is_empty_leaf(self, node):
        """
        Check if a node is an empty leaf.

        :param node: The node to check.
        :return: True if the node is an empty leaf, False otherwise.
        """
        return self.is_leaf(node) and not self.leaf_samples[node]
