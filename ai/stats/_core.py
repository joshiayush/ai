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
# pylint: disable=too-many-function-args, invalid-name, missing-module-docstring
# pylint: disable=missing-class-docstring

from typing import Optional

import numpy as np


def _validate_array_dims(func) -> callable:
  def wrapper(*args, **kwargs):
    for arg in args:
      if isinstance(arg, np.ndarray) and arg.ndim > 2:
        raise ValueError(
          f'{func.__name__} does not support vector of more than 2 dimensions.'
        )

    if kwargs['axis'] and kwargs['axis'] > 2:
      raise ValueError(
        f'{func.__name__} does not support vector of more than 2 dimensions.'
      )
    return func(*args, **kwargs)

  return wrapper


def _transpose_vector_if_required(func) -> callable:
  def wrapper(*args, **kwargs):
    args_t = tuple()
    for arg in args:
      if isinstance(arg, np.ndarray):
        if 'axis' in kwargs and kwargs['axis'] is None:
          args_t += (np.ravel(arg), )
        elif ('axis' in kwargs and kwargs['axis'] == 1):
          args_t += (arg.T, )
        else:
          args_t += (arg, )
    return func(*args_t, **kwargs)

  return wrapper


def _mean(x: np.ndarray) -> np.float64:
  return np.sum(x) / x.size


@_validate_array_dims
@_transpose_vector_if_required
def mean(a: np.ndarray, /, *, axis: Optional[int] = None) -> np.ndarray:
  mean = []
  if axis is None:
    mean.append(_mean(a))
  else:
    for x in a:
      mean.append(_mean(x))
  return np.array(mean)


def _median(x: np.ndarray) -> np.float64:
  x_sorted = np.sort(x, kind='stable')
  mid_idx = x_sorted.size // 2
  if x_sorted.size % 2 != 0:
    return x_sorted[mid_idx]
  else:
    return np.divide(x_sorted[mid_idx] + x_sorted[-1 * mid_idx], 2)


@_validate_array_dims
@_transpose_vector_if_required
def median(a: np.ndarray, /, *, axis: Optional[int] = None) -> np.ndarray:
  median = []
  if axis is None:
    median.append(_median(a))
  else:
    for x in a:
      median.append(_median(x))
  return np.array(median)


def _std(x: np.ndarray, /, *, ddof: Optional[int] = 1) -> np.float64:
  n = x.size
  return np.sqrt(
    np.divide(
      n * np.sum(np.power(x, 2)) - np.power(np.sum(x), 2), n * (n - ddof)
    )
  )


@_validate_array_dims
@_transpose_vector_if_required
def std(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  std = []
  if ddof not in (
    0,
    1,
  ):
    raise ValueError(f'ddof must be in (0, 1), you gave ddof={ddof}')
  if axis is None:
    std.append(_std(a, ddof=ddof))
  else:
    for x in a:
      std.append(_std(x, ddof=ddof))
  return np.array(std)


@_validate_array_dims
@_transpose_vector_if_required
def var(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  if axis == 1:
    # if array is already transposed we prevent it from further transpose in the
    # next call since (X_T)_T = X
    axis = 0
  return np.power(std(a, ddof=ddof, axis=axis), 2)


def _zscore(
  x: np.ndarray, /, *, mean: np.float64, std: np.float64
) -> np.float64:
  return np.divide(x - mean, std)


@_validate_array_dims
@_transpose_vector_if_required
def zscore(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  if axis == 1:
    # if array is already transposed we prevent it from further transpose in the
    # next call since (X_T)_T = X
    axis = 0
  zscore = []
  a_mean = mean(a, axis=axis)
  a_std = std(a, ddof=ddof, axis=axis)
  if axis is None:
    zscore.append(_zscore(a, mean=a_mean, std=a_std))
  else:
    for idx, x in enumerate(a):
      zscore.append(_zscore(x, mean=a_mean[idx], std=a_std[idx]))
  return np.array(zscore)


@_validate_array_dims
@_transpose_vector_if_required
def varcoef(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  if axis == 1:
    # if array is already transposed we prevent it from further transpose in the
    # next call since (X_T)_T = X
    axis = 0
  return np.divide(std(a, ddof=ddof, axis=axis), mean(a, axis=axis)) * 100
