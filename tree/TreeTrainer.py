import numpy as np

from tree.Tree import Tree


class TreeTrainer:
    def __init__(self, relabeling_strategy, splitting_rule_factory, prediction_strategy):
        """
        Initialize a TreeTrainer.

        :param relabeling_strategy: Strategy for relabeling nodes.
        :param splitting_rule_factory: Factory for creating splitting rules.
        :param prediction_strategy: Strategy for making predictions.
        """
        self.relabeling_strategy = relabeling_strategy
        self.splitting_rule_factory = splitting_rule_factory
        self.prediction_strategy = prediction_strategy

    def train(self, data, sampler, clusters, options):
        """
        Train a decision tree.

        :param data: The data used for training the tree.
        :param sampler: A RandomSampler instance.
        :param clusters: Clusters of data.
        :param options: Tree options.
        :return: A trained Tree object.
        """
        child_nodes = [[], []]
        nodes = [[]]
        split_vars = []
        split_values = []
        send_missing_left = []

        self.create_empty_node(child_nodes, nodes, split_vars, split_values, send_missing_left)

        new_leaf_samples = []
        if options.get_honesty():
            tree_growing_clusters, new_leaf_clusters = sampler.subsample(clusters, options.get_honesty_fraction())
            nodes[0] = sampler.sample_from_clusters(tree_growing_clusters)
            new_leaf_samples = sampler.sample_from_clusters(new_leaf_clusters)
        else:
            nodes[0] = sampler.sample_from_clusters(clusters)

        splitting_rule = self.splitting_rule_factory.create(len(nodes[0]), options)

        num_open_nodes = 1
        i = 0
        responses_by_sample = np.zeros((data.get_num_rows(), self.relabeling_strategy.get_response_length()))
        while num_open_nodes > 0:
            is_leaf_node = self.split_node(i, data, splitting_rule, sampler, child_nodes, nodes, split_vars, split_values, send_missing_left, responses_by_sample, options)
            if is_leaf_node:
                num_open_nodes -= 1
            else:
                nodes[i].clear()
                num_open_nodes += 1
            i += 1

        drawn_samples = sampler.get_samples_in_clusters(clusters)
        tree = Tree(0, child_nodes, nodes, split_vars, split_values, drawn_samples, send_missing_left, PredictionValues())

        if new_leaf_samples:
            self.repopulate_leaf_nodes(tree, data, new_leaf_samples, options.get_honesty_prune_leaves())

        prediction_values = PredictionValues()  # Placeholder for actual prediction values
        if self.prediction_strategy:
            prediction_values = self.prediction_strategy.precompute_prediction_values(tree.get_leaf_samples(), data)
        tree.set_prediction_values(prediction_values)

        return tree


    def repopulate_leaf_nodes(self, tree, data, leaf_samples, honesty_prune_leaves):
        """
        Repopulate the leaf nodes of the tree with new samples.

        :param tree: The tree to repopulate.
        :param data: The data used for the tree.
        :param leaf_samples: The new leaf samples.
        :param honesty_prune_leaves: Whether to apply honesty-based pruning.
        """
        num_nodes = len(tree.get_leaf_samples())
        new_leaf_nodes = [[] for _ in range(num_nodes)]

        leaf_nodes = tree.find_leaf_nodes(data, leaf_samples)

        for sample in leaf_samples:
            leaf_node = leaf_nodes[sample]
            new_leaf_nodes[leaf_node].append(sample)

        tree.set_leaf_samples(new_leaf_nodes)

        if honesty_prune_leaves:
            tree.honesty_prune_leaves()

    def create_split_variable_subset(self, sampler, data, mtry):
        """
        Create a subset of variables for splitting.

        :param sampler: A RandomSampler instance.
        :param data: The data used for training the tree.
        :param mtry: The number of variables to try at each split.
        :return: A list of variable indices to be used for splitting.
        """
        num_independent_variables = data.get_num_cols() - len(data.get_disallowed_split_variables())
        mtry_sample = sampler.sample_poisson(mtry)
        split_mtry = max(min(mtry_sample, num_independent_variables), 1)

        return sampler.draw(data.get_num_cols(), data.get_disallowed_split_variables(), split_mtry)

    def split_node(self, node, data, splitting_rule, sampler, child_nodes, samples, split_vars, split_values, send_missing_left, responses_by_sample, options):
        """
        Split a node during the tree training process.

        :param node: The index of the current node to be split.
        :param data: The data used for training the tree.
        :param splitting_rule: The rule used for splitting nodes.
        :param sampler: A RandomSampler instance.
        :param child_nodes: A list of child nodes.
        :param samples: A list of samples at each node.
        :param split_vars: Variables used for splitting at each node.
        :param split_values: Values used for splitting at each node.
        :param send_missing_left: Whether to send missing values left at each node.
        :param responses_by_sample: The responses associated with each sample.
        :param options: Tree options.
        :return: True if the node is a terminal node, False otherwise.
        """
        possible_split_vars = self.create_split_variable_subset(sampler, data, options.get_mtry())

        # Placeholder for split_node_internal function
        stop = split_node_internal(node, data, splitting_rule, possible_split_vars, samples, split_vars, split_values, send_missing_left, responses_by_sample, options.get_min_node_size())

        if stop:
            return True

        split_var = split_vars[node]
        split_value = split_values[node]
        send_na_left = send_missing_left[node]

        left_child_node = len(samples)
        child_nodes[0][node] = left_child_node
        self.create_empty_node(child_nodes, samples, split_vars, split_values, send_missing_left)

        right_child_node = len(samples)
        child_nodes[1][node] = right_child_node
        self.create_empty_node(child_nodes, samples, split_vars, split_values, send_missing_left)

        for sample in samples[node]:
            value = data.get(sample, split_var)
            if value <= split_value or (send_na_left and math.isnan(value)) or (math.isnan(split_value) and math.isnan(value)):
                samples[left_child_node].append(sample)
            else:
                samples[right_child_node].append(sample)

        return False

    def split_node_internal(self, node, data, splitting_rule, possible_split_vars, samples, split_vars, split_values, send_missing_left, responses_by_sample, min_node_size):
        """
        Split a node during the tree training process.

        :param node: The index of the current node to be split.
        :param data: The data used for training the tree.
        :param splitting_rule: The rule used for splitting nodes.
        :param possible_split_vars: A list of possible variables for splitting.
        :param samples: A list of samples at each node.
        :param split_vars: Variables used for splitting at each node.
        :param split_values: Values used for splitting at each node.
        :param send_missing_left: Whether to send missing values left at each node.
        :param responses_by_sample: The responses associated with each sample.
        :param min_node_size: The minimum size of a node.
        :return: True if the node is a terminal node, False otherwise.
        """
        if len(samples[node]) <= min_node_size:
            split_values[node] = -1.0
            return True

        stop = self.relabeling_strategy.relabel(samples[node], data, responses_by_sample)

        if stop or splitting_rule.find_best_split(data, node, possible_split_vars, responses_by_sample, samples, split_vars, split_values, send_missing_left):
            split_values[node] = -1.0
            return True

        return False

    def create_empty_node(self, child_nodes, samples, split_vars, split_values, send_missing_left):
        """
        Create an empty node in the tree with default values.

        :param child_nodes: A list of child nodes.
        :param samples: A list of samples at each node.
        :param split_vars: Variables used for splitting at each node.
        :param split_values: Values used for splitting at each node.
        :param send_missing_left: Whether to send missing values left at each node.
        """
        child_nodes[0].append(0)
        child_nodes[1].append(0)
        samples.append([])
        split_vars.append(0)
        split_values.append(0.0)
        send_missing_left.append(True)

