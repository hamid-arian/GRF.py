from typing import List, Union
from tree import Tree  # assuming Tree is defined in a separate file


class Forest:
    def __init__(self, trees: List[Tree], num_variables: int, ci_group_size: int):
        self.trees = list(trees)
        self.num_variables = num_variables
        self.ci_group_size = ci_group_size

    @property
    def get_trees(self):
        return self.trees

    @property
    def get_num_variables(self):
        return self.num_variables

    @property
    def get_ci_group_size(self):
        return self.ci_group_size

    @staticmethod
    def merge(forests: List['Forest']) -> 'Forest':
        all_trees = []
        num_variables = forests[0].get_num_variables()
        ci_group_size = forests[0].get_ci_group_size()

        for forest in forests:
            trees = forest.get_trees
            all_trees.extend(trees)

            if forest.get_ci_group_size() != ci_group_size:
                raise RuntimeError("All forests being merged must have the same ci_group_size.")

        return Forest(all_trees, num_variables, ci_group_size)
