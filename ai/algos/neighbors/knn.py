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

from typing import Union

import numpy as np

from collections import Counter


class DistanceMetric:
  _distance_func_cache = None

  def __init__(self, metric: str, minkowski_p: int = 2):
    self._metric = metric
    self._minkowski_p = minkowski_p

  def _euclidean(
    self, x1: Union[np.float32, np.ndarray], x2: Union[np.float32, np.ndarray]
  ) -> Union[np.float32, np.ndarray]:
    return np.sqrt(np.power(np.sum((x1 - x2)), 2))

  def _minkowski(
    self, x1: Union[np.float32, np.ndarray], x2: Union[np.float32, np.ndarray]
  ) -> Union[np.float32, np.ndarray]:
    return np.power(np.sum(np.absolute((x1 - x2))), 1 / self._minkowski_p)

  def _manhattan(
    self, x1: Union[np.float32, np.ndarray], x2: Union[np.float32, np.ndarray]
  ) -> Union[np.float32, np.ndarray]:
    return np.sum(np.absolute((x1 - x2)))

  def _hamming(
    self, x1: Union[np.float32, np.ndarray], x2: Union[np.float32, np.ndarray]
  ) -> Union[np.float32, np.ndarray]:
    return np.sum(np.absolute((x1 - x2)))

  def distance(
    self, x1: Union[np.float32, np.ndarray], x2: Union[np.float32, np.ndarray]
  ) -> Union[np.float32, np.ndarray]:
    if self._distance_func_cache is not None:
      return self._distance_func_cache(x1, x2)

    if self._metric == 'euclidean':
      self._distance_func_cache = self._euclidean
    elif self._metric == 'minkowski':
      self._distance_func_cache = self._minkowski
    elif self._metric == 'manhattan':
      self._distance_func_cache = self._manhattan
    elif self._metric == 'hamming':
      self._distance_func_cache = self._hamming
    else:
      raise RuntimeError(
        (
          f'{self.__class__.__name__}: {self._metric} is not one of ["euclidean",'
          ' "minkowski", "manhattan", "hamming"]'
        )
      )
    return self._distance_func_cache(x1, x2)


class KNeighborsClassifier(DistanceMetric):
  _parameter_constraints: dict = {
    'metric': [
      ('euclidean', 'supported'), ('minkowski', 'not-supported'),
      ('manhattan', 'not-supported'), ('hamming', 'not-supported')
    ]
  }

  @staticmethod
  def _check_if_parameters_comply_to_constraints(**params: dict) -> None:
    is_distance_metric_present = False
    for (metric_name, metric_status
         ) in KNeighborsClassifier._parameter_constraints['metric']:
      if params['metric'] == metric_name:
        is_distance_metric_present = True

      if is_distance_metric_present is True:
        if metric_status != 'supported':
          raise RuntimeError(
            f'distance metric {metric_name} is not supported yet'
          )
        break

  def __init__(
    self, *, n_neighbors: int = 3, p: int = 2, metric: str = 'euclidean'
  ):
    self._n_neighbors = n_neighbors
    self._p = p
    self._metric = metric
    self._fit_on_dataset = False

    self._check_if_parameters_comply_to_constraints(metric=self._metric)
    super().__init__(self._metric, self._p)

  def fit(self, X: np.ndarray, y: np.ndarray):
    self._X = X
    self._y = y
    self._fit_on_dataset = True

  def predict(self, X: np.ndarray) -> np.ndarray:
    if self._fit_on_dataset is False:
      raise RuntimeError(
        f'{self.__class__.__name__}: predict called before fitting data'
      )

    preds = []
    for x in X:
      distances = [self.distance(x, x_train) for x_train in self._X]

      k_indices = np.argsort(distances)[:self._n_neighbors]
      k_nearest_labels = [self.y_train[i] for i in k_indices]

      preds = [*preds, Counter(k_nearest_labels).most_common()[0][0]]
    return preds
