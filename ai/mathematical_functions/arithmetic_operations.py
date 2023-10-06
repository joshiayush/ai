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

from typing import Union, Optional

import numpy as np


def proportion(
  x1: Union[np.float32, np.ndarray],
  x2: Union[np.float32, np.ndarray],
  y1: Union[np.float32, np.ndarray],
  y2: Union[np.float32, np.ndarray],
  *,
  solve_for: Optional[str] = None
) -> None:
  """Solves for either `[x1, x2, y1, y2]`, when they are `zeros`.

  This is the implementation of the **arithmetic proportion algorithm** that
  solves for either the numerator or the denominator from either side given the
  other values.

  .. math::

    \\dfrac{x_{1}}{y_{1}} = \\dfrac{x_{2}}{y_{2}}

  This function modifies the values of the `zero` vector in-place.

  .. note::

    In cases of more than one possible `zero_like` numpy vector, you must
    specify explicitly for the variable the function must solve for. Otherwise,
    the first found variable that is `zero` will be assumed to solve for.

    The order in which the algorithm decides for which variable to solve for is:
      * `x1`
      * `x2`
      * `y1`
      * `y2`

  Args:
    x1: NumPy vector for the left side numerator.
    x2: NumPy vector for the right side numerator.
    y1: NumPy vector for the left side denominator.
    y2: NumPy vector for the right side denominator.

  Raises:
    ValueError: If neither of `[x1, x2, y1, y2]` are `zeros`.
    ZeroDivisionError: If the denominator in the final equation is `zero`.

  Examples:

  >>> # Example 1: Solve for x1 when x2, y1, and y2 are known
  >>> x1 = np.zeros(3)
  >>> x2 = np.array([3.0, 4.0, 5.0])
  >>> y1 = np.array([6.0, 7.0, 8.0])
  >>> y2 = np.array([9.0, 10.0, 11.0])
  >>> proportion(x1, x2, y1, y2, solve_for='x1')
  >>> x1
  ... array([2., 2.8, 3.63636364])
  """
  if solve_for is None:
    if np.all(x1 == 0):
      solve_for = 'x1'
    elif np.all(x2 == 0):
      solve_for = 'x2'
    elif np.all(y1 == 0):
      solve_for = 'y1'
    elif np.all(y2 == 0):
      solve_for = 'y2'

  if solve_for is None:
    raise ValueError(
      (
        'Atleast numerator or denominator must be all zeros from either side to'
        ' calculate proportion.'
      )
    )

  if solve_for == 'x1':
    if np.any(y2 == 0):
      raise ZeroDivisionError('The denominator `y2` cannot be zero.')
    x1[...] = np.divide(np.multiply(x2, y1), y2)
  elif solve_for == 'x2':
    if np.any(y1 == 0):
      raise ZeroDivisionError('The denominator `y1` cannot be zero.')
    x2[...] = np.divide(np.multiply(x1, y2), y1)
  elif solve_for == 'y1':
    if np.any(x2 == 0):
      raise ZeroDivisionError('The denominator `x2` cannot be zero.')
    y1[...] = np.divide(np.multiply(x1, y2), x2)
  elif solve_for == 'y2':
    if np.any(x1 == 0):
      raise ZeroDivisionError('The denominator `x1` cannot be zero.')
    y2[...] = np.divide(np.multiply(x2, y1), x1)
