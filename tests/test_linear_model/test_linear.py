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

from ai.linear_model import LinearRegression


def test_linear_regression_fit():
  lr = LinearRegression()
  X = np.array([[1.], [2.], [3.]])
  y = np.array([2., 4., 6.])
  lr.fit(X, y)
  assert lr._is_fitted


def test_linear_regression_predict():
  lr = LinearRegression()
  X = np.array([[1.], [2.], [3.]])
  y = np.array([2., 4., 6.])
  lr.fit(X, y)
  y_pred = lr.predict(X)  # Predicting on the `X` sample vector for now
  assert np.allclose(y_pred, y, rtol=1111111e-7)


def test_linear_regression_predict_before_fit():
  lr = LinearRegression()
  X = np.array([[1], [2], [3]])
  with pytest.raises(RuntimeError):
    lr.predict(X)


def test_linear_regression_fit_shape_mismatch():
  lr = LinearRegression()
  X = np.array([[1], [2], [3]])
  y = np.array([2, 4])
  with pytest.raises(ValueError):
    lr.fit(X, y)
