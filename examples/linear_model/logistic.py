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

import matplotlib.pyplot as plt
import numpy as np

from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from ai import linear_model

# Generate a toy dataset, it's just a straight line with some Gaussian noise:
xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)
y = (X > 0).astype(float)
X[X > 0] *= 4
X += 0.3 * np.random.normal(size=n_samples)

X = X[:, np.newaxis]

print('Estimation using sklearn api...')

# Fit the classifier
clf = LogisticRegression(C=1e5)
clf.fit(X, y)

# and plot the result
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X.ravel(), y, label="example data", color="black", zorder=20)
X_test = np.linspace(-5, 10, 300)

loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(
  X_test,
  loss,
  label="Logistic Regression Model (sklearn)",
  color="red",
  linewidth=3
)

ols = LinearRegression()
ols.fit(X, y)
plt.plot(
  X_test,
  ols.coef_ * X_test + ols.intercept_,
  label="Linear Regression Model",
  linewidth=1,
)
plt.axhline(0.5, color=".5")

plt.ylabel("y")
plt.xlabel("X")
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-0.25, 1.25)
plt.xlim(-4, 10)
plt.legend(
  loc="lower right",
  fontsize="small",
)
plt.tight_layout()
plt.show()

print('Estimation using ai api...')
print(
  "Since there's no implementation of the closed-form solution for linear"
  " regression in ai; sklearn api beats us in terms of speed..."
)

# Fit the classifier
# LogisticRegression internally uses LinearRegression for making the hypothesis,
# so wait for a while before the estimation pops up
clf = linear_model.LogisticRegression(n_iters=1_000_000)
clf.fit(X, y)

# and plot the result
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X.ravel(), y, label="example data", color="black", zorder=20)
X_test = np.linspace(-5, 10, 300)

loss = expit(X_test * clf._weights + clf._bias).ravel()
plt.plot(
  X_test,
  loss,
  label="Logistic Regression Model (ai)",
  color="red",
  linewidth=3
)

ols = LinearRegression()
ols.fit(X, y)
plt.plot(
  X_test,
  ols.coef_ * X_test + ols.intercept_,
  label="Linear Regression Model",
  linewidth=1,
)
plt.axhline(0.5, color=".5")

plt.ylabel("y")
plt.xlabel("X")
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-0.25, 1.25)
plt.xlim(-4, 10)
plt.legend(
  loc="lower right",
  fontsize="small",
)
plt.tight_layout()
plt.show()
