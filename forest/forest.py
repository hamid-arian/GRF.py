class Forest:
    def __init__(self, trees, num_variables, ci_group_size):
        """
        Initialize a Forest.

        :param trees: A list of Tree objects.
        :param num_variables: The number of variables in the dataset.
        :param ci_group_size: The group size used for confidence interval calculation.
        """
        self.trees = trees
        self.num_variables = num_variables
        self.ci_group_size = ci_group_size

    @classmethod
    def merge(cls, forests):
        """
        Merge multiple forests into one.

        :param forests: A list of Forest objects to be merged.
        :return: A new merged Forest.
        """
        if any(forest.ci_group_size != forests[0].ci_group_size for forest in forests):
            raise ValueError("All forests being merged must have the same ci_group_size.")

        num_variables = forests[0].num_variables
        ci_group_size = forests[0].ci_group_size
        all_trees = [tree for forest in forests for tree in forest.trees]

        return cls(all_trees, num_variables, ci_group_size)

    def get_trees(self):
        """
        Get the trees in the forest.

        :return: A list of Tree objects.
        """
        return self.trees

    def get_num_variables(self):
        """
        Get the number of variables in the dataset.

        :return: The number of variables.
        """
        return self.num_variables

    def get_ci_group_size(self):
        """
        Get the group size used for confidence interval calculation.

        :return: The group size.
        """
        return self.ci_group_size
