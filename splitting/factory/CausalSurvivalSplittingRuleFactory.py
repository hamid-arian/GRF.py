"""
This file is part of generalized random forest (grf).

grf is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

grf is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with grf. If not, see <http://www.gnu.org/licenses/>.
"""
from splitting.factory.CausalSurvivalSplittingRuleFactory import CausalSurvivalSplittingRuleFactory
from splitting.CausalSurvivalSplittingRule import CausalSurvivalSplittingRule

class GrfNamespace:
    @staticmethod
    def create(max_num_unique_values, options):
        return CausalSurvivalSplittingRuleFactory.create(
            max_num_unique_values,
            options.get_min_node_size(),
            options.get_alpha(),
            options.get_imbalance_penalty()
        )