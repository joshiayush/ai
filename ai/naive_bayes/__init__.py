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

"""Naive Bayes methods are a set of `supervised` learning algorithms based on
applying `Bayes’` theorem with the “naive” assumption of conditional
independence between every pair of features given the value of the class
variable. Bayes theorem states the following relationship, given class variable
`y` and dependent feature vector :math:`x_{1}` through :math:`x_{n}`:

.. math::

    P(y | x_{1}, ..., x_{n}) = \\dfrac{
                                  P(y) \\cdot P(x_{1} ..., x_{n} | y)
                                }{
                                  P(x_{1} ..., x_{n})
                                }

Using the naive conditional independence assumption that:

.. math::

    P(x_{i} | y, x_{1}, ..., x_{i-1}, x_{i+1}, ..., x_{n}) = P(x_{i} | y)

for all :math:`i`, this relationship is simplified to:

.. math::

    P(y | x_{1}, ..., x_{n}) = \\dfrac{
                                  P(y) \\cdot \\prod_{i=1}^{n}P(x_{i} | y)
                                }{
                                  P(x_{1} ..., x_{n})
                                }

Since :math:`P(x_{1}, ..., x_{n})` is constant given the input, we can use the
following classification rule:

.. math::

    P(y | x_{1}, ..., x_{n}) \\propto P(y) \\cdot \\prod_{i=1}^{n}P(x_{i} | y)

.. math::

    ⇒ \\hat y = arg \\max_{y} P(y) \\cdot \\prod_{i=1}^{n} P(x_{i} | y)

and we can use **Maximum A Posteriori (MAP)** estimation to estimate
:math:`P(y)` and :math:`P(x_{i} | y)`;  the former is then the relative
frequency of class :math:`y` in the training set.

`ai.naive_bayes` implements the following `naive bayes` algorithms:

  * `ai.naive_bayes.naive_bayes.GaussianNaiveBayes`
"""

from .naive_bayes import GaussianNaiveBayes
