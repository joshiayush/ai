# Copyright 2023 The AI Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np

from ai.mathematical_functions.arithmetic_operations import proportion


def test_proportion_auto_detection():
  x1 = np.array([0.0, 0.0, 0.0])
  x2 = np.array([3.0, 4.0, 5.0])
  y1 = np.array([6.0, 7.0, 8.0])
  y2 = np.array([9.0, 10.0, 11.0])
  proportion(x1, x2, y1, y2)
  # Check if x1 has been modified in-place
  assert np.all(x1 != [0.0, 1.0, 2.0])


def test_proportion_explicit_solve():
  x1 = np.array([0.0, 0.0, 0.0])
  x2 = np.array([3.0, 4.0, 5.0])
  y1 = np.array([6.0, 7.0, 8.0])
  y2 = np.array([9.0, 10.0, 11.0])
  proportion(x1, x2, y1, y2, solve_for='x1')
  # Check if x1 has been modified in-place
  assert np.all(x1 != [0.0, 1.0, 2.0])


def test_proportion_no_zeros():
  x1 = np.array([1.0, 2.0, 3.0])
  x2 = np.array([4.0, 5.0, 6.0])
  y1 = np.array([7.0, 8.0, 9.0])
  y2 = np.array([10.0, 11.0, 12.0])
  with pytest.raises(ValueError):
    proportion(x1, x2, y1, y2)


def test_proportion_calculation():
  x1 = np.array([0.0, 1.0, 2.0])
  x2 = np.array([3.0, 4.0, 5.0])
  y1 = np.array([6.0, 7.0, 8.0])
  y2 = np.array([9.0, 10.0, 11.0])
  proportion(x1, x2, y1, y2, solve_for='x1')
  # Check if x1 has been correctly calculated based on the proportion
  assert np.allclose(x1, [2., 2.8, 3.63636364])
