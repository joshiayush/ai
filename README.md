# Artificial Intelligence Guide

These are the source files for the guide on Artificial Intelligence.

To contribute to the Artificial Intelligence Guide, please read the
[style guide](https://www.tensorflow.org/community/contribute/docs_style).

To jump onto the __Artificial Intelligence__ roadmap, please click on the
[Artificial Intelligence Map](https://github.com/joshiayush/ai/tree/master/docs/roadmap.md).

## Contents

1. [Resources](https://github.com/joshiayush/ai/tree/master/docs/resource) (Resources required during learning)
2. [Prework](https://github.com/joshiayush/ai/tree/master/docs/prework) (A bit of prework required before starting machine learning)
3. [Tools](https://github.com/joshiayush/ai/tree/master/docs/tools) (Tools required to build machine learning models)
4. [Machine Learning](https://github.com/joshiayush/ai/tree/master/docs/ml) (A deep dive into the concepts of machine learning)
5. [Algorithms](https://github.com/joshiayush/ai/tree/master/docs/algos) (Implementation of common machine learning algorithms)

## API Usage

Using the `LinearRegression` estimator:

```python
import numpy as np

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ai.linear_model import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_pred, y_test))
```
