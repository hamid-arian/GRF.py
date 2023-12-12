import random
import math
import numpy as np

class SamplingOptions:
    def __init__(self, samples_per_cluster=0, clusters=None):
        if clusters is None:
            clusters = []
        self.num_samples_per_cluster = samples_per_cluster
        self.clusters = clusters

    def get_clusters(self):
        return self.clusters

    def get_samples_per_cluster(self):
        return self.num_samples_per_cluster


class RandomSampler:
    def __init__(self, seed, options):
        self.random_number_generator = random.Random(seed)
        self.options = options

    def sample_clusters(self, num_rows, sample_fraction, samples):
        if not self.options.get_clusters():
            self.sample(num_rows, sample_fraction, samples)
        else:
            num_samples = len(self.options.get_clusters())
            self.sample(num_samples, sample_fraction, samples)

    def sample_from_clusters(self, clusters, samples):
        if not self.options.get_clusters():
            samples.extend(clusters)
        else:
            samples_by_cluster = self.options.get_clusters()
            for cluster in clusters:
                cluster_samples = samples_by_cluster[cluster]
                if len(cluster_samples) <= self.options.get_samples_per_cluster():
                    samples.extend(cluster_samples)
                else:
                    subsamples = self.subsample_with_size(cluster_samples, self.options.get_samples_per_cluster())
                    samples.extend(subsamples)

    def get_samples_in_clusters(self, clusters, samples):
        if not self.options.get_clusters():
            samples.extend(clusters)
        else:
            for cluster in clusters:
                cluster_samples = self.options.get_clusters()[cluster]
                samples.extend(cluster_samples)

    def sample(self, num_samples, sample_fraction, samples):
        num_samples_inbag = int(num_samples * sample_fraction)
        self.shuffle_and_split(samples, num_samples, num_samples_inbag)

    def subsample(self, samples, sample_fraction, subsamples, oob_samples=None):
        shuffled_sample = np.random.permutation(samples)

        subsample_size = math.ceil(len(samples) * sample_fraction)
        subsamples.extend(shuffled_sample[:subsample_size])

        if oob_samples is not None:
            oob_samples.extend(shuffled_sample[subsample_size:])

    def subsample_with_size(self, samples, subsample_size):
        shuffled_sample = np.random.permutation(samples)
        return shuffled_sample[:subsample_size]

    def shuffle_and_split(self, samples, n_all, size):
        samples.extend(np.random.permutation(np.arange(n_all))[:size])

    def draw(self, result, max_val, skip, num_samples):
        if num_samples < max_val / 10:
            self.draw_simple(result, max_val, skip, num_samples)
        else:
            self.draw_fisher_yates(result, max_val, skip, num_samples)

    def draw_simple(self, result, max_val, skip, num_samples):
        result.extend(np.random.choice([i for i in range(max_val) if i not in skip], size=num_samples, replace=False))

    def draw_fisher_yates(self, result, max_val, skip, num_samples):
        result.extend(np.setdiff1d(np.random.permutation(max_val), list(skip))[:num_samples])

    def sample_poisson(self, mean):
        return np.random.poisson(mean)


# Example usage:
seed_value = 42
sampling_options = SamplingOptions(samples_per_cluster=3, clusters=None)
random_sampler = RandomSampler(seed=seed_value, options=sampling_options)

# Call the necessary methods on random_sampler
# For example:
# random_sampler.sample_clusters(100, 0.5, samples)
# random_sampler.sample_from_clusters(clusters, samples)
# random_sampler.get_samples_in_clusters(clusters, samples)
# random_sampler.subsample(samples, 0.5, subsamples, oob_samples)
# random_sampler.draw(result, max_val, skip, num_samples)
# random_sampler.sample_poisson(mean)
