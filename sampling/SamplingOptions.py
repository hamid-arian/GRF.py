class SamplingOptions:
    """
    A class to handle sampling options, particularly for cluster-based sampling.
    """

    def __init__(self, samples_per_cluster=0, sample_clusters=None):
        """
        Initialize SamplingOptions.

        :param samples_per_cluster: The number of samples per cluster.
        :param sample_clusters: A list of sample clusters.
        """
        self.num_samples_per_cluster = samples_per_cluster

        if sample_clusters is None:
            self.clusters = []
        else:
            # Map the provided clusters to IDs in the range 0 ... num_clusters
            cluster_ids = {}
            for cluster in sample_clusters:
                if cluster not in cluster_ids:
                    cluster_ids[cluster] = len(cluster_ids)

            # Populate the index of each cluster ID with the samples it contains
            self.clusters = [[] for _ in range(len(cluster_ids))]
            for sample, cluster in enumerate(sample_clusters):
                cluster_id = cluster_ids[cluster]
                self.clusters[cluster_id].append(sample)

    def get_samples_per_cluster(self):
        """
        Get the number of samples per cluster.

        :return: The number of samples per cluster.
        """
        return self.num_samples_per_cluster

    def get_clusters(self):
        """
        Get the clusters.

        :return: The clusters as a list of lists.
        """
        return self.clusters
