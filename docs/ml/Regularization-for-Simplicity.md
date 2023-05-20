# Regularization for Simplicity

__Regularization__ means penalizing the complexity of a model to reduce overfitting.

## L2 Regularization

Consider the following __generalization curve__, which shows the loss for both the training set and validation set against the number of training iterations.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/RegularizationTwoLossFunctions.svg' />

  <strong>Figure 1. Loss on training set and validation set.</strong>
</div>

Figure 1 shows a model in which training loss gradually decreases, but validation loss eventually goes up. In other words, this generalization curve shows that the model is [overfitting](https://developers.google.com/machine-learning/crash-course/generalization/peril-of-overfitting) to the data in the training set. Channeling our inner [Ockham](https://developers.google.com/machine-learning/crash-course/generalization/peril-of-overfitting#ockham), perhaps we could prevent overfitting by penalizing complex models, a principle called __regularization__.

In other words, instead of simply aiming to minimize loss (empirical risk minimization):

$$\mathrm{minimize(Loss(Data \vert Model))}$$

we'll now minimize loss+complexity, which is called __structural risk minimization__.

$$\mathrm{minimize(Loss(Data \vert Model) + \mathrm{complexity(Model)})}$$

Our training optimization algorithm is now a function of two terms: the __loss term__, which measures how well the model fits the data, and the __regularization term__, which measures model complexity.

There are two commom ways to think of model complexity:

  * Model complexity as a function of the _weights_ of all the features in the model.
  * Model complexity as a function of the _total number of features_ with nonzero weights.

If model complexity is a function of weights, a feature weight with a high absolute value is more complex that a feature weight with a low absolute value.

We can quantify complexity using the $L_{2}$ __regularization__ formula, which defines the regularization term as the sum of the squares of all the feature weights:

$$\mathrm{L_{2}\ regularization\ term} = \mathrm{\lvert \lvert w \rvert \rvert}^2_{2} = w_{1}^2 + w_{2}^2 + ... + w_{n}^2$$

In this formula, weights close to zero have little effect on model complexity, while outlier weights can have a huge impact.

For example, a linear model with the following weights:

$$\{w_{1} = 0.2, w_{2} = 0.5, w_{3} = 5, w_{4} = 1, w_{5} = 0.25, w_{6} = 0.75\}$$

Has an $L_{2}$ regularization term of 26.915:

```math
w_{1}^2 + w_{2}^2 + w_{3}^2 + w_{4}^2 + w_{5}^2 + w_{6}^2 \\
= 0.2^2 + 0.5^2 + 5^2 + 1^2 + 0.25^2 + 0.75^2 \\
= 0.04 + 0.24 + 25 + 1 + 0.0625 + 0.5625 \\
= 26.915
```

But $w_{3}$, with a squared value of 25, contributes nearly all the complexity. The sum of the squares of all five other weights adds just 1.915 to the $L_{2}$ regularization term.