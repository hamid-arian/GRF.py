class SplitFrequencyComputer:
    def compute(self, forest, max_depth):
        """
        Compute the frequency of splits for each variable at different depths in a forest.

        :param forest: The Forest object.
        :param max_depth: The maximum depth to which the trees should be traversed.
        :return: A 2D list where each sublist represents a depth, and each element in the sublist 
                 is the count of splits for a variable at that depth.
        """
        num_variables = forest.get_num_variables()
        result = [[0 for _ in range(num_variables)] for _ in range(max_depth)]

        for tree in forest.get_trees():
            child_nodes = tree.get_child_nodes()

            depth = 0
            level = [tree.get_root_node()]

            while len(level) > 0 and depth < max_depth:
                next_level = []

                for node in level:
                    if tree.is_leaf(node):
                        continue

                    variable = tree.get_split_vars()[node]
                    result[depth][variable] += 1

                    next_level.append(child_nodes[0][node])
                    next_level.append(child_nodes[1][node])

                level = next_level
                depth += 1

        return result


