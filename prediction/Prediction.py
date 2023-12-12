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

class Prediction:
    def __init__(self, predictions, variance_estimates=None, error_estimates=None, excess_error_estimates=None):
        self.predictions = predictions
        self.variance_estimates = variance_estimates if variance_estimates is not None else []
        self.error_estimates = error_estimates if error_estimates is not None else []
        self.excess_error_estimates = excess_error_estimates if excess_error_estimates is not None else []

    def get_predictions(self):
        return self.predictions

    def get_variance_estimates(self):
        return self.variance_estimates

    def get_error_estimates(self):
        return self.error_estimates

    def get_excess_error_estimates(self):
        return self.excess_error_estimates

    def contains_variance_estimates(self):
        return bool(self.variance_estimates)

    def contains_error_estimates(self):
        return bool(self.error_estimates)

    def size(self):
        return len(self.predictions)
