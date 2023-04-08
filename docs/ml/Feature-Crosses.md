# Feature Crosses

A __feature cross__ is a __synthetic feature__ formed by multiplying (crossing) two or more features. Crossing combinations of features can provide predictive abilities beyond what those features can provide individually.

## Encoding Nonlinearity

In Figures 1 and 2, imagine the following:

* The blue dots represent sick trees.
* The orange dots represent healthy trees.

<div align='center'>
  <img src='https://developers.google.com/machine-learning/crash-course/images/LinearProblem1.png' />

  <strong>Figure 1. Is this a linear problem?</strong>
</div>

Can you draw a line that neatly separates the sick trees from the healthy trees? Sure. This is a linear problem. The line won't be perfect. A sick tree or two might be on the "healthy" side, but your line will be a good predictor.

Now look at the following figure:

<div align='center'>
  <img src='https://developers.google.com/machine-learning/crash-course/images/LinearProblem2.png' />

  <strong>Figure 2. Is this a linear problem?</strong>
</div>

Can you draw a single straight line that neatly separates the sick trees from the healthy trees? No, you can't. This is a nonlinear problem. Any line you draw will be a poor predictor of tree health.

<div align='center'>
  <img src='https://developers.google.com/machine-learning/crash-course/images/LinearProblemNot.png' />

  <strong>Figure 3. A single line can't separate the two classes.</strong>
</div>

To solve the nonlinear problem shown in Figure 2, create a feature cross. A __feature cross__ is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together. (The term cross comes from [cross product](https://wikipedia.org/wiki/Cross_product).) Let's create a feature cross named $x_{3}$ by crossing $x_{1}$ and $x_{2}$:

$$x_{3}=x_{1}x_{2}$$

We treat this newly minted $x_{3}$ feature cross just like any other feature. The linear formula becomes:

$$y=b+w_{1}x_{1}+w_{2}x_{2}+w_{3}x_{3}$$

A linear algorithm can learn a weight for $w_{3}$ just as it would for $w_{1}$ and $w_{2}$. In other words, although $w_{3}$ encodes nonlinear information, you don’t need to change how the linear model trains to determine the value of $w_{3}$.

### Kinds of feature crosses

We can create many different kinds of feature crosses. For example:

* `[A X B]`: a feature cross formed by multiplying the values of two features.
* `[A x B x C x D x E]`: a feature cross formed by multiplying the values of five features.
* `[A x A]`: a feature cross formed by squaring a single feature.

Thanks to [stochastic gradient descent](https://developers.google.com/machine-learning/crash-course/reducing-loss/stochastic-gradient-descent), linear models can be trained efficiently. Consequently, supplementing scaled linear models with feature crosses has traditionally been an efficient way to train on massive-scale data sets.