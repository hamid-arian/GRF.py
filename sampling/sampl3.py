from typing import List
from collections import defaultdict

class SamplingOptions:
    def __init__(self, samples_per_cluster=0, sample_clusters=None):
        self.num_samples_per_cluster = samples_per_cluster
        self.clusters = []
        if sample_clusters is not None:
            self.map_clusters(sample_clusters)

    def map_clusters(self, sample_clusters):
        cluster_ids = {}
        for cluster in sample_clusters:
            if cluster not in cluster_ids:
                cluster_id = len(cluster_ids)
                cluster_ids[cluster] = cluster_id

        self.clusters = [[] for _ in range(len(cluster_ids))]
        for sample, cluster in enumerate(sample_clusters):
            cluster_id = cluster_ids[cluster]
            self.clusters[cluster_id].append(sample)

    def get_samples_per_cluster(self):
        return self.num_samples_per_cluster

    def get_clusters(self):
        return self.clusters

