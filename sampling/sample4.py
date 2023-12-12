from typing import List

class SamplingOptions:
    def __init__(self, samples_per_cluster=0, clusters=None):
        self.num_samples_per_cluster = samples_per_cluster
        self.clusters = clusters or []

    def get_clusters(self):
        return self.clusters

    def get_samples_per_cluster(self):
        return self.num_samples_per_cluster
