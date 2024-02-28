import math
import random

class RandomSampler:
    """
    A class for random sampling and subsampling of data.
    """

    def __init__(self, seed, options):
        """
        Initialize the RandomSampler with a seed and options.

        :param seed: The seed for the random number generator.
        :param options: An object containing sampling options.
        """
        self.random_number_generator = random.Random(seed)
        self.options = options

    def sample_clusters(self, num_rows, sample_fraction, samples):
        """
        Sample clusters of data.

        :param num_rows: The number of rows in the data.
        :param sample_fraction: The fraction of data to sample.
        :param samples: A list to store the sampled indices.
        """
        if not self.options.get_clusters():
            self.sample(num_rows, sample_fraction, samples)
        else:
            num_samples = len(self.options.get_clusters())
            self.sample(num_samples, sample_fraction, samples)

    def sample(self, num_samples, sample_fraction, samples):
        """
        Sample a fraction of data.

        :param num_samples: The number of samples to draw.
        :param sample_fraction: The fraction of data to sample.
        :param samples: A list to store the sampled indices.
        """
        num_samples_inbag = int(num_samples * sample_fraction)
        samples.extend(self.random_number_generator.sample(range(num_samples), num_samples_inbag))

    def subsample(self, samples, sample_fraction, subsamples, oob_samples=None):
        """
        Subsample a fraction of the provided samples.

        :param samples: The samples to subsample from.
        :param sample_fraction: The fraction of data to subsample.
        :param subsamples: A list to store the subsampled indices.
        :param oob_samples: A list to store out-of-bag samples if needed.
        """

        shuffled_sample = samples
        self.random_number_generator.shuffle(shuffled_sample)

        subsample_size = int(math.ceil(len(samples) * sample_fraction))
        subsamples.extend(shuffled_sample[:subsample_size])

        if oob_samples is not None:
            oob_samples.extend(shuffled_sample[subsample_size:])

    def subsample_with_size(self, samples, subsample_size, subsamples):
        """
        Subsample a specific number of samples.

        :param samples: The samples to subsample from.
        :param subsample_size: The number of samples to draw.
        :param subsamples: A list to store the subsampled indices.
        """
        shuffled_sample = samples[:]
        self.random_number_generator.shuffle(shuffled_sample)
        subsamples.extend(shuffled_sample[:subsample_size])

    def sample_from_clusters(self, clusters, samples):
        """
        Sample from clusters.

        :param clusters: The cluster indices.
        :param samples: A list to store the sampled indices.
        """
        if not self.options.get_clusters():
            samples.extend(clusters)
        else:
            samples_by_cluster = self.options.get_clusters()
            for cluster in clusters:
                cluster_samples = samples_by_cluster[cluster]

                if len(cluster_samples) <= self.options.get_samples_per_cluster():
                    samples.extend(cluster_samples)
                else:
                    subsamples = []
                    self.subsample_with_size(cluster_samples, self.options.get_samples_per_cluster(), subsamples)
                    samples.extend(subsamples)

    def get_samples_in_clusters(self, clusters, samples):
        """
        Get samples in clusters.

        :param clusters: The cluster indices.
        :param samples: A list to store the samples from the clusters.
        """
        if not self.options.get_clusters():
            samples.extend(clusters)
        else:
            for cluster in clusters:
                cluster_samples = self.options.get_clusters()[cluster]
                samples.extend(cluster_samples)

    def shuffle_and_split(self, n_all, size):
        """
        Shuffle and split the samples.

        :param n_all: The total number of samples.
        :param size: The size of the desired sample.
        :return: A list of shuffled and split samples.
        """
        samples = list(range(n_all))
        self.random_number_generator.shuffle(samples)
        return samples[:size]

    def draw(self, max, skip, num_samples):
        """
        Draw samples, choosing the method based on the number of samples.

        :param max: The maximum value for drawing.
        :param skip: A set of values to skip.
        :param num_samples: The number of samples to draw.
        :return: A list of drawn samples.
        """
        if num_samples < max / 10:
            return self.draw_simple(max, skip, num_samples)
        else:
            return self.draw_fisher_yates(max, skip, num_samples)

    def draw_simple(self, max, skip, num_samples):
        """
        Draw samples using a simple method.

        :param max: The maximum value for drawing.
        :param skip: A set of values to skip.
        :param num_samples: The number of samples to draw.
        :return: A list of drawn samples.
        """
        result = []
        temp = [False] * max

        while len(result) < num_samples:
            draw = self.random_number_generator.randint(0, max - 1 - len(skip))
            for skip_value in sorted(skip):
                if draw >= skip_value:
                    draw += 1
            if not temp[draw]:
                temp[draw] = True
                result.append(draw)

        return result

    def draw_fisher_yates(self, max, skip, num_samples):
        """
        Draw samples using the Fisher-Yates shuffle algorithm.

        :param max: The maximum value for drawing.
        :param skip: A set of values to skip.
        :param num_samples: The number of samples to draw.
        :return: A list of drawn samples.
        """
        result = list(range(max))
        for skip_value in sorted(skip, reverse=True):
            result.pop(skip_value)

        for i in range(num_samples):
            j = self.random_number_generator.randint(i, max - len(skip) - 1)
            result[i], result[j] = result[j], result[i]

        return result[:num_samples]

    def sample_poisson(self, mean):
        """
        Sample from a Poisson distribution.

        :param mean: The mean of the Poisson distribution.
        :return: A random sample from the Poisson distribution.
        """
        return self.random_number_generator.poisson(mean)

