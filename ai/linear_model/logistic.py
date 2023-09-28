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

import numpy as np


class LogisticRegression:
  """Logistic Regression (aka logit) classifier.

  The logistic regression model transforms the linear regression function
  continuous value output into categorical value output using a sigmoid
  function, which maps any real-valued set of independent variables input into
  a value between 0 and 1. This function is known as the logistic function.

  .. math::

    z = w \\cdot X + b

  Now we use the sigmoid function where the input will be z and we find the
  probability between 0 and 1. i.e predicted y.

  .. math::

    \\sigma (z) = \\dfrac{1}{1 - e^{-z}}

  Args:
    alpha: Model's learning rate. High value might over shoot the minimum
      loss, while low values might make the model to take forever to learn.
    n_iters: Maximum number of updations to make over the weights and bias in
      order to reach to a effecient prediction that minimizes the loss.
  """
  def __init__(self, alpha: np.float16 = .01, n_iters: np.int64 = 1000):
    """Initializes model's `learning rate` and number of `iterations`.

    Args:
      alpha: Model's learning rate. High value might over shoot the minimum
        loss, while low values might make the model to take forever to learn.
      n_iters: Maximum number of updations to make over the weights and bias in
        order to reach to a effecient prediction that minimizes the loss.
    """
    self._alpha = alpha
    self._n_iters = n_iters

    self._bias = None
    self._weights = None

  @staticmethod
  def _sigmoid(t: np.ndarray) -> np.ndarray:
    """Sigmoid function to find the probability of `t` between 0 and 1.

    Args:
      t: Model predictions.

    Returns:
      A value between 0 and 1 based on the sigmoid function.
    """
    return 1 / (1 + np.exp(-t))

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
    """Fit Logistic Regression according to X, y.

    The logistic regression model transforms the linear regression function
    continuous value output into categorical value output using a sigmoid
    function, which maps any real-valued set of independent variables input into
    a value between 0 and 1. This function is known as the logistic function.

    .. math::

      z = w \\cdot X + b

    Now we use the sigmoid function where the input will be z and we find the
    probability between 0 and 1. i.e predicted y.

    .. math::

      \\sigma (z) = \\dfrac{1}{1 - e^{-z}}

    Args:
      X: Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.
      y: Target values.

    Returns:
      Returns the instance itself.
    """
    self._bias = 0
    self._weights = np.zeros(X.shape[1])

    for _ in range(self._n_iters):
      y_pred = self._sigmoid(np.dot(X, self._weights) + self._bias)

      bias_d = 1 / X[0] * np.sum((y_pred - y))
      weights_d = 1 / X[0] * np.dot(X.T, (y_pred - y))

      self._bias = self._bias - (self._alpha * bias_d)
      self._weights = self._weights - (self._alpha * weights_d)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict for `X` using the previously calculated `weights` and `bias`.

    Args:
      X: Feature vector.

    Returns:
      Target vector.

    Raises:
      RuntimeError: If `predict` is called before `fit`.
      ValueError: If shape of the given `X` differs from the shape of the `X`
        given to the `fit` function.
    """
    if self._weights is None or self._bias is None:
      raise RuntimeError(
        f'{self.__class__.__name__}: predict called before fitting data'
      )

    if X.shape[1] != self._weights.shape[0]:
      raise ValueError(
        (
          f'Number of features {X.shape[1]} does not match previous data '
          f'{self._weights.shape[0]}.'
        )
      )

    y_pred = self._sigmoid(np.dot(X, self._weights) + self._bias)
    c_pred = [0 if y <= .5 else 1 for y in y_pred]
    return c_pred