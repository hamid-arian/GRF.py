from typing import List, Dict

class SamplingOptions:
    def __init__(self, samples_per_cluster: int = 0, sample_clusters: List[int] = None):
        if sample_clusters is None:
            sample_clusters = []
        
        self.num_samples_per_cluster = samples_per_cluster
        self.clusters = {}
        
        cluster_ids = {}
        for cluster in sample_clusters:
            if cluster not in cluster_ids:
                cluster_id = len(cluster_ids)
                cluster_ids[cluster] = cluster_id
        
        for sample, cluster in enumerate(sample_clusters):
            cluster_id = cluster_ids[cluster]
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(sample)

    def get_samples_per_cluster(self) -> int:
        return self.num_samples_per_cluster

    def get_clusters(self) -> Dict[int, List[int]]:
        return self.clusters


# Example usage:
# sampling_options = SamplingOptions(samples_per_cluster=3, sample_clusters=[0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 2, 2, 2, 2, 0, 3, 3, 3, 2, 3])
# print(sampling_options.get_samples_per_cluster())
# print(sampling_options.get_clusters())
