# forest.py

from tree import Tree

class Forest:
    def __init__(self, trees, num_variables, ci_group_size):
        self.trees = list(trees)
        self.num_variables = num_variables
        self.ci_group_size = ci_group_size

    def get_trees(self):
        return self.trees

    def get_trees_(self):
        return self.trees

    def get_num_variables(self):
        return self.num_variables

    def get_ci_group_size(self):
        return self.ci_group_size

    @staticmethod
    def merge(forests):
        all_trees = []
        num_variables = forests[0].get_num_variables()
        ci_group_size = forests[0].get_ci_group_size()

        for forest in forests:
            trees = forest.get_trees_()
            all_trees.extend(trees)

            if forest.get_ci_group_size() != ci_group_size:
                raise RuntimeError("All forests being merged must have the same ci_group_size.")

        return Forest(all_trees, num_variables, ci_group_size)
