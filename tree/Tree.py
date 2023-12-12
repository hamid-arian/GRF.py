class Tree:
    def __init__(self, root_node, child_nodes, leaf_samples, split_vars, split_values, drawn_samples, send_missing_left, prediction_values):
        self.root_node = root_node
        self.child_nodes = child_nodes
        self.leaf_samples = leaf_samples
        self.split_vars = split_vars
        self.split_values = split_values
        self.drawn_samples = drawn_samples
        self.send_missing_left = send_missing_left
        self.prediction_values = prediction_values

    def find_leaf_nodes(self, data, samples):
        prediction_leaf_nodes = [0] * data.get_num_rows()
        for sample in samples:
            node = self.find_leaf_node(data, sample)
            prediction_leaf_nodes[sample] = node
        return prediction_leaf_nodes

    # Additional methods will follow a similar pattern

