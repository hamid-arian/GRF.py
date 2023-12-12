# This file is part of generalized random forest (grf).
#
# grf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# grf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with grf. If not, see <http://www.gnu.org/licenses/>.

class ProbabilityPredictionStrategy:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.num_types = num_classes + 1
        self.weight_index = num_classes

    def prediction_length(self):
        return self.num_classes

    def predict(self, average):
        weight_bar = average[self.weight_index]
        predictions = [average[cls] / weight_bar for cls in range(self.num_classes)]
        return predictions

    def compute_variance(self, average, leaf_values, ci_group_size):
        variance_estimates = [0] * self.num_classes
        weight_bar = average[self.weight_index]

        for cls in range(self.num_classes):
            average_outcome = average[cls] / weight_bar

            num_good_groups = 0
            rho_squared = 0
            rho_grouped_squared = 0

            for group in range(leaf_values.get_num_nodes() // ci_group_size):
                good_group = True

                for j in range(ci_group_size):
                    if leaf_values.empty(group * ci_group_size + j):
                        good_group = False

                if not good_group:
                    continue

                num_good_groups += 1
                group_rho = 0

                for j in range(ci_group_size):
                    i = group * ci_group_size + j
                    rho = (leaf_values.get(i, cls) - average_outcome * leaf_values.get(i, self.weight_index)) / weight_bar
                    rho_squared += rho * rho
                    group_rho += rho

                group_rho /= ci_group_size
                rho_grouped_squared += group_rho * group_rho

            var_between = rho_grouped_squared / num_good_groups
            var_total = rho_squared / (num_good_groups * ci_group_size)

            group_noise = (var_total - var_between) / (ci_group_size - 1)
            var_debiased = self.bayes_debiaser.debias(var_between, group_noise, num_good_groups)
            variance_estimates[cls] = var_debiased

        return variance_estimates

    def prediction_value_length(self):
        return self.num_types

    def precompute_prediction_values(self, leaf_samples, data):
        num_leaves = len(leaf_samples)
        values = []

        for i in range(num_leaves):
            leaf_node = leaf_samples[i]

            if not leaf_node:
                continue

            averages = [0] * self.num_types
            weight_sum = 0.0

            for sample in leaf_node:
                sample_class = int(data.get_outcome(sample))
                averages[sample_class] += data.get_weight(sample)
                weight_sum += data.get_weight(sample)

            if abs(weight_sum) <= 1e-16:
                averages.clear()
                continue

            for cls in range(self.num_classes):
                averages[cls] = averages[cls] / len(leaf_node)

            averages[self.weight_index] = weight_sum / len(leaf_node)
            values.append(averages)

        return PredictionValues(values, self.num_types)

    def compute_error(self, sample, average, leaf_values, data):
        return [(float('nan'), float('nan'))]
