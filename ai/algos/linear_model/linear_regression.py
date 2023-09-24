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


class LinearRegression:
  """`LinearRegression` fits a linear model with coefficients w = (w1, ..., wp)
  to minimize the residual sum of squares between the observed targets in the
  dataset, and the targets predicted by the linear approximation.

  Hypothesis function for our `LinearRegression`: math:`\\hat y = b + wX`, where
  `b` is the model's intercept and `w` is the coefficient of `X`.

  The cost function or the loss function that we use is the Mean Squared Error
  (MSE) between the predicted value and the true value. The cost function `(J)`
  can be written as:

    math:`J = \\dfrac{1}{m}\\sum_{i=1}^{n}(\\hat y_{i} - y_{i})^2`

  To achieve the best-fit regression line, the model aims to predict the target
  value math:`\\hat Y` such that the error difference between the predicted
  value math:`\\hat Y` and the true value math:`Y` is minimum. So, it is very
  important to update the `b` and `w` values, to reach the best value that
  minimizes the error between the predicted `y` value and the true `y` value.

  A linear regression model can be trained using the optimization algorithm
  gradient descent by iteratively modifying the model’s parameters to reduce the
  mean squared error (MSE) of the model on a training dataset. To update `b` and
  `w` values in order to reduce the Cost function (minimizing RMSE value) and
  achieve the best-fit line the model uses Gradient Descent. The idea is to
  start with random `b` and `w` values and then iteratively update the values,
  reaching minimum cost.

  On differentiating cost function `J` with respect to `b`:

    math:`
    \\dfrac{dJ}{db} = \\dfrac{2}{n} \\cdot \\sum_{i=1}^{n}(
                      \\hat y_{i} - y_{i}
                      )
    `

  On differentiating cost function `J` with respect to `w`:

    math:`
    \\dfrac{dJ}{dw} = \\dfrac{2}{n} \\cdot \\sum_{i=1}^{n}(
                      \\hat y_{i} - y_{i}
                      ) \\cdot x_{i}
    `

  The above derivative functions are used for updating `weights` and `bias` in
  each iteration.

  Args:
    lr: Model's learning rate. High value might over shoot the minimum loss,
      while low values might make the model to take forever to learn.
    n_iters: Maximum number of updations to make over the weights and bias in
      order to reach to a effecient prediction that minimizes the loss.
  """
  def __init__(self, *, lr: np.float16 = .01, n_iters: int = 1000):
    """Initializes model's `learning rate` and number of `iterations`.

    Args:
      lr: Model's learning rate. High value might over shoot the minimum loss,
        while low values might make the model to take forever to learn.
      n_iters: Maximum number of updations to make over the weights and bias in
        order to reach to a effecient prediction that minimizes the loss.
    """
    self._lr = lr
    self._n_iters = n_iters
    self._is_fitted = False

  def fit(self, X: np.ndarray, y: np.ndarray) -> None:
    """Fit the linear model on `X` given `y`.

    Args:
      X: Sample vector.
      y: Target vector.

    Hypothesis function for our `LinearRegression`: math:`\\hat y = b + wX`,
    where `b` is the model's intercept and `w` is the coefficient of `X`.

    The cost function or the loss function that we use is the Mean Squared Error
    (MSE) between the predicted value and the true value. The cost function
    `(J)` can be written as:

      math:`J = \\dfrac{1}{m}\\sum_{i=1}^{n}(\\hat y_{i} - y_{i})^2`

    To achieve the best-fit regression line, the model aims to predict the
    target value math:`\\hat Y` such that the error difference between the
    predicted value math:`\\hat Y` and the true value math:`Y` is minimum. So,
    it is very important to update the `b` and `w` values, to reach the best
    value that minimizes the error between the predicted `y` value and the true
    `y` value.

    A linear regression model can be trained using the optimization algorithm
    gradient descent by iteratively modifying the model’s parameters to reduce
    the mean squared error (MSE) of the model on a training dataset. To update
    `b` and `w` values in order to reduce the Cost function (minimizing RMSE
    value) and achieve the best-fit line the model uses Gradient Descent. The
    idea is to start with random `b` and `w` values and then iteratively update
    the values, reaching minimum cost.

    On differentiating cost function `J` with respect to `b`:

      math:`
      \\dfrac{dJ}{db} = \\dfrac{2}{n} \\cdot \\sum_{i=1}^{n}(
                        \\hat y_{i} - y_{i}
                        )
      `

    On differentiating cost function `J` with respect to `w`:

      math:`
      \\dfrac{dJ}{dw} = \\dfrac{2}{n} \\cdot \\sum_{i=1}^{n}(
                        \\hat y_{i} - y_{i}
                        ) \\cdot x_{i}
      `

    The above derivative functions are used for updating `weights` and `bias` in
    each iteration.
    """
    self._bias = 0
    self._weights = np.zeros(X[1])

    for _ in range(self._n_iters):
      y_pred = np.dot(X, self._weights) + self._bias

      bias_d = 1 / X[0] * np.sum((y_pred - y))
      weights_d = 1 / X[0] * np.dot(X.T, (y_pred - y))

      self._bias = self._bias - (self._lr * bias_d)
      self._weights = self._weights - (self._lr * weights_d)

    self._is_fitted = True

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
    if self._is_fitted is False:
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

    return np.dot(X, self._weights) + self._bias
