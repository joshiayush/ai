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
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from ai import LinearRegression

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object sklearn api
model = linear_model.LinearRegression()

# Train the model using the training set
model.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = model.predict(diabetes_X_test)

print("Calculating using sklearn api...")

# The coefficients
print("(sklearn) Coefficients: \n", model.coef_)
# The mean squared error
print(
  "(sklearn) Mean squared error: %.2f" %
  mean_squared_error(diabetes_y_test, diabetes_y_pred)
)
# The coefficient of determination: 1 is perfect prediction
print(
  "(sklearn) Coefficient of determination: %.2f" %
  r2_score(diabetes_y_test, diabetes_y_pred)
)

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

print("Calculating using ai api...")
print(
  "Since there's no implementation of the closed-form solution for linear"
  " regression in ai; sklearn api beats us in terms of speed..."
)

# Create linear regression object using ai api
# Since there's no implementation of the closed-form solution for linear
# regression in `ai`; `sklearn` api beats us in terms of speed
model = LinearRegression(n_iters=1_000_000)

# Train the model using the training set
model.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = model.predict(diabetes_X_test)

# The coefficients
print("(ai) Coefficients: \n", model._weights)
# The mean squared error
print(
  "(ai) Mean squared error: %.2f" %
  mean_squared_error(diabetes_y_test, diabetes_y_pred)
)
# The coefficient of determination: 1 is perfect prediction
print(
  "(ai) Coefficient of determination: %.2f" %
  r2_score(diabetes_y_test, diabetes_y_pred)
)

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
