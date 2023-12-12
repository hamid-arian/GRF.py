
class SplitFrequencyComputer:
    def compute(self, forest, max_depth):
        num_variables = forest.get_num_variables()
        result = [[0 for _ in range(num_variables)] for _ in range(max_depth)]

        for tree in forest.get_trees():
            child_nodes = tree.get_child_nodes()

            depth = 0
            level = [tree.get_root_node()]

            while level and depth < max_depth:
                next_level = []

                for node in level:
                    if tree.is_leaf(node):
                        continue

                    variable = tree.get_split_vars()[node]
                    result[depth][variable] += 1

                    next_level.extend(child_nodes[0][node])
                    next_level.extend(child_nodes[1][node])

                level = next_level
                depth += 1

        return result

# Example Usage
# forest = ... # Initialize your Forest object
# computer = SplitFrequencyComputer()
# split_frequencies = computer.compute(forest, max_depth)
