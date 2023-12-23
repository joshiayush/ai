# Changelog

## ai — 2023-12-23

### Added

- Added `statistic` calculating functions to `ai.stats` module (#20) ([#6c8e592](https://www.github.com/joshiayush/ai/commit/6c8e592))


## ai — 2023-12-09

### Added

- Added tests for `LogisticRegression` ([#6aef58a](https://www.github.com/joshiayush/ai/commit/6aef58a))
- Added documentation to `Not` class that represents the `¬` (not) symbol in a propositional logic (PL) ([#429f1f1](https://www.github.com/joshiayush/ai/commit/429f1f1))
- Added examples comparing both `sklearn` and `ai` api's `LinearRegression` estimator over a single feature of `diabetes` dataset ([#098ce8c](https://www.github.com/joshiayush/ai/commit/098ce8c))
- Added example comparing `ai.linear_model.LogisticRegression` estimator against `sklearn.linear_model.LogisticRegression` ([#e7b8a92](https://www.github.com/joshiayush/ai/commit/e7b8a92))

### Fixed

- Fixed calculation for `weights` and `bias` in logistic regression ([#d156644](https://www.github.com/joshiayush/ai/commit/d156644))
- Fixed logical error in calculating the derivatives of `weights` and `bias` in linear regression ([#aa46cfe](https://www.github.com/joshiayush/ai/commit/aa46cfe))

### Removed

- Removed mathematical function `proportion` which serves no use for `ai` ([#6fe7d8c](https://www.github.com/joshiayush/ai/commit/6fe7d8c))

## docs — 2023-12-09

### Added

- Added concept of `Linear Unit`, `Layers`, `Stacking Dense Layers`, and `Dropout and Batch Normalization` to `Neural Networks` ([#64337d7](https://www.github.com/joshiayush/ai/commit/64337d7))
- Added a __Support Vector Machine__ model trained for face recognition ([#912c4c2](https://www.github.com/joshiayush/ai/commit/912c4c2))

### Fixed

- Fixed in-consistent aspect ration problem ([#365509d](https://www.github.com/joshiayush/ai/commit/365509d))


## ai — 2023-11-22

### Added

- Added search utility frontiers and boolean algebra for representing propositional logic (PL) ([#04df291](https://www.github.com/joshiayush/ai/commit/04df291))
- Added light weight numpy-versatile implementation of the `Linear Support Vector Machines Classifier` (#13) ([#fbecf95](https://www.github.com/joshiayush/ai/commit/fbecf95))
- Added an implementation of `Gaussian Radial Basis Function` kernel for SVMs (#17) ([#b8c57d8](https://www.github.com/joshiayush/ai/commit/b8c57d8))
- Added `NumPy-Compatible` mathematical function `proportion` to solve for the missing value given the proportion equation ([#928376f](https://www.github.com/joshiayush/ai/commit/928376f))
- Added a brief explaination of `Training Neural Networks` into the main `ai` documentation ([#26ba4bf](https://www.github.com/joshiayush/ai/commit/26ba4bf))

## docs — 2023-11-22

### Added

- Added a brief explaination of `Training Neural Networks` into the main `ai` documentation ([#26ba4bf](https://www.github.com/joshiayush/ai/commit/26ba4bf))
- Added `Transforming Categorical Data` by mapping categories to feature vectors ([#2473732](https://www.github.com/joshiayush/ai/commit/2473732))
- Added `Introduction to Transforming Your Data` with `Normalization` and `Bucketing` (Only for Numerical Data) ([#cf1740f](https://www.github.com/joshiayush/ai/commit/cf1740f))
- Added best practices to train a neural network ([#ed7ab7a](https://www.github.com/joshiayush/ai/commit/ed7ab7a))

### Fixed

- Fixed formatting errors related to text rendering and `LaTeX` blocks (#12) ([#88366d1](https://www.github.com/joshiayush/ai/commit/88366d1))

### Removed

- Removed "Introduction to Tensorflow" to keep the document focused only on the theoretical part ([#feafdee](https://www.github.com/joshiayush/ai/commit/feafdee))


## ai — 2023-09-30

### Added

- Added a light weight implementation of `KNeighborsClassifier` classification algorithm using pure `numpy` ([#6f2e6fc](https://www.github.com/joshiayush/ai/commit/6f2e6fc))
- Added detailed explaination of `Machine Learning` methods and concepts into the main `ai` documentation ([#2936fb9](https://www.github.com/joshiayush/ai/commit/2936fb9))
- Added `Accessing array rows and columns`, `Subarrays as no-copy views`, `Creating copies of arrays`, `Reshaping of arrays` ([#c547776](https://www.github.com/joshiayush/ai/commit/c547776))
- Added documentation to submodules landing page ([#2821baf](https://www.github.com/joshiayush/ai/commit/2821baf))
- Added a light weight implementation of `GaussianNaiveBayes` classification algorithm using `numpy` and `gaussian` distribution approach ([#b1ed8bf](https://www.github.com/joshiayush/ai/commit/b1ed8bf))
- Added `Gaussian Naive Bayes` classifier trained on `iris` dataset ([#ed032b9](https://www.github.com/joshiayush/ai/commit/ed032b9))
- Added light weight implementation of `LogisticRegression` (aka logit) classifier ([#66408c9](https://www.github.com/joshiayush/ai/commit/66408c9))
- Added tests for `LinearRegression` `fit` and `predict` method ([#4a3cfd6](https://www.github.com/joshiayush/ai/commit/4a3cfd6))
- Added `ROC (receiver operating characteristic curve) Curve and AUC (area under curve)` section ([#9148452](https://www.github.com/joshiayush/ai/commit/9148452))
- Added light weight implementation of `LinearRegression` using the `Gradient Descent` optimization function ([#7567f14](https://www.github.com/joshiayush/ai/commit/7567f14))


### Fixed

- Fixed docstring syntax errors ([#28aa72d](https://www.github.com/joshiayush/ai/commit/28aa72d))
- Fixed the roadmap image un-responsive problem ([#bc27fb2](https://www.github.com/joshiayush/ai/commit/bc27fb2))


## docs — 2023-09-30

### Added

- Added detailed explaination of `Machine Learning` methods and concepts into the main `ai` documentation ([#2936fb9](https://www.github.com/joshiayush/ai/commit/2936fb9))
- Added `README` to every sub-module level for better doc navigation ([#f14d8f3](https://www.github.com/joshiayush/ai/commit/f14d8f3))


### Fixed

- Fixed bug -- Un-neccessary addition of "ebooks" directory as a parent directory for the documents ([#81dd279](https://www.github.com/joshiayush/ai/commit/81dd279))