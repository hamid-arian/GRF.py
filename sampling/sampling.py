import numpy as np
import random

class RandomSampler:
    def __init__(self, seed, options):
        self.options = options
        random.seed(seed)

    def sample_clusters(self, num_rows, sample_fraction):
        samples = []
        if not self.options.get_clusters():
            self.sample(num_rows, sample_fraction, samples)
        else:
            num_samples = len(self.options.get_clusters())
            self.sample(num_samples, sample_fraction, samples)
        return samples

    def sample(self, num_samples, sample_fraction, samples):
        num_samples_inbag = int(num_samples * sample_fraction)
        self.shuffle_and_split(samples, num_samples, num_samples_inbag)

    def subsample(self, samples, sample_fraction):
        shuffled_sample = list(samples)
        random.shuffle(shuffled_sample)

        subsample_size = int(len(samples) * sample_fraction)
        subsamples = shuffled_sample[:subsample_size]
        return subsamples

    def subsample_with_size(self, samples, subsample_size):
        shuffled_sample = list(samples)
        random.shuffle(shuffled_sample)

        subsamples = shuffled_sample[:subsample_size]
        return subsamples

    def sample_from_clusters(self, clusters):
        samples = []
        if not self.options.get_clusters():
            samples = clusters
        else:
            samples_by_cluster = self.options.get_clusters()
            for cluster in clusters:
                cluster_samples = samples_by_cluster[cluster]

                if len(cluster_samples) <= self.options.get_samples_per_cluster():
                    samples.extend(cluster_samples)
                else:
                    subsamples = self.subsample_with_size(cluster_samples, self.options.get_samples_per_cluster())
                    samples.extend(subsamples)
        return samples

    def get_samples_in_clusters(self, clusters):
        samples = []
        if not self.options.get_clusters():
            samples = clusters
        else:
            for cluster in clusters:
                cluster_samples = self.options.get_clusters()[cluster]
                samples.extend(cluster_samples)
        return samples

    def shuffle_and_split(self, samples, n_all, size):
        samples.extend(random.sample(range(n_all), n_all))
        samples = samples[:size]

    def draw(self, max_value, skip, num_samples):
        result = []
        if num_samples < max_value / 10:
            self.draw_simple(result, max_value, skip, num_samples)
        else:
            self.draw_fisher_yates(result, max_value, skip, num_samples)
        return result

    def draw_simple(self, result, max_value, skip, num_samples):
        temp = [False] * max_value

        for i in range(num_samples):
            draw = random.randint(0, max_value - 1 - len(skip))
            for skip_value in skip:
                if draw >= skip_value:
                    draw += 1
            while temp[draw]:
                draw = random.randint(0, max_value - 1 - len(skip))
                for skip_value in skip:
                    if draw >= skip_value:
                        draw += 1
            temp[draw] = True
            result.append(draw)

    def draw_fisher_yates(self, result, max_value, skip, num_samples):
        result.extend(range(max_value))
        for skip_value in sorted(skip, reverse=True):
            result.pop(skip_value)

        for i in range(num_samples):
            j = int(i + random.random() * (max_value - len(skip) - i))
            result[i], result[j] = result[j], result[i]

        result = result[:num_samples]

    def sample_poisson(self, mean):
        return np.random.poisson(mean)
