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

import math
from commons.utility import equal_doubles

class ObjectiveBayesDebiaser:
    def debias(self, var_between, group_noise, num_good_groups):
        # Let S denote the true between-groups variance, and assume that
        # group_noise is measured exactly; our method-of-moments estimate is
        # then \hat{S} = var_between - group_noise. Now, if we take
        # num_good_groups * var_between to be chi-squared with scale
        # S + group_noise and num_good_groups degrees of freedom, and assume
        # that group_noise >> S, we find that var[initial_estimate] is roughly
        # var_between^2 * 2 / num_good_groups; moreover, the distribution of
        # \hat{S} - S is roughly Gaussian with this variance. Our estimation strategy
        # relies on this fact, and puts a uniform prior on S for the interval [0, infty).
        # This debiasing does nothing when \hat{S} >> var_between * sqrt(2 / num_good_groups),
        # but keeps \hat{S} from going negative.

        initial_estimate = var_between - group_noise
        initial_se = max(var_between, group_noise) * math.sqrt(2.0 / num_good_groups)
        if equal_doubles(initial_se, 0.0, 1.0e-10):
            return 0.0

        ratio = initial_estimate / initial_se

        # corresponds to \int_(-r)^infty x * phi(x) dx, for the standard Gaussian density phi(x)
        numerator = math.exp(- ratio * ratio / 2) * math.sqrt(2 / math.pi)

        # corresponds to int_(-r)^infty phi(x) dx
        denominator = 0.5 * math.erfc(-ratio / math.sqrt(2))

        # this is the o-Bayes estimate of the error of the initial estimate
        bayes_correction = initial_se * numerator / denominator
        return initial_estimate + bayes_correction
