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

class PredictionValues:
    def __init__(self):
        self.num_nodes = 0
        self.num_types = 0
        self.values = []

    def __init__(self, values, num_types):
        self.values = values
        self.num_nodes = len(values)
        self.num_types = num_types

    def get(self, node, type):
        return self.values[node][type]

    def get_values(self, node):
        return self.values[node]

    def empty(self, node):
        return not self.values[node]

    def get_all_values(self):
        return self.values

    def get_num_nodes(self):
        return self.num_nodes

    def get_num_types(self):
        return self.num_types
