# Machine Learning

# Introduction to ML

Machine learning is a field of inquiry devoted to understanding and building methods that __'learn'__, that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence.

## Additional information

* _Rules of Machine Learning,_ [Rule #1: Don't be afraid to launch a product without machine learning](https://developers.google.com/machine-learning/rules-of-ml/#rule_1_dont_be_afraid_to_launch_a_product_without_machine_learning)

## Framing: Key ML Terminology

What is (supervised) machine learning? Concisely put, it is the following:

* ML systems learn how to combine input to produce useful predictions on never-before-seen data.

Let's explore fundamental machine learning terminology.

## Labels

A __label__ is the thing we are predicting - the `y` variable in simple linear regression. The label could be the future price of wheat, the kind of animal shown in the picture, the meaning of an audio clip, or just about anything.

## Features

A __feature__ is an input variable - the `x` variable in simple linera regression. A simple machine learning project might use a single feature, while a more sophisticated machine learning project could use millions of features, specified as:

$$x1,x2,...,xn$$

In the "spam detector" example, the features could include the following:

* words in the email text
* sender's address
* time of day the email was sent
* email contains the phrase "one weird trick."

## Examples

An __example__ is a particular instance of data, __x__. (We put __x__ in boldface to indicate that it is a vector.) We break __examples__ into two categories:

* labeled examples
* unlabeled examples

A __labeled example__ includes both feature(s) and the label. That is:

```
labeled examples: {features, label}: (x, y)
```

Use __labeled examples__ to train the model. In our "span detector" example, the labeled examples would be individual emails that users have explicitly marked as "spam" or "not spam."

For example, the following table shows 5 labeled examples from a data set containing information about housing prices in California:

__HousingMedianAge (feature)__ | __TotalRooms (feature)__ | __TotalBedrooms (feature)__ | __MedianHouseValue (feature)__
:--|:--:|:--:|--:
15 | 5612 | 1283 | 66900
19 | 7650 | 1901 | 80100
17 | 720 | 174 | 85700
14 |	1501 |	337	| 73400
20	| 1454 |	326 |	65500

An __unlabeled example__ contains feature(s) but not the label. That is:

```
unlabeled examples: {features, ?}: (x, ?)
```

Here are 3 unlabeled examples from the same housing dataset, which exclude __`MedianHouseValue`__:

__HousingMedianAge (feature)__ |	__TotalRooms (feature)__ |	__TotalBedrooms (feature)__
:--|:--:|--:
42 |	1686 |	361
34 |	1226 |	180
33 |	1077 |	271

Once we've trained our model with labeled examples, we use that model to predict the label on unlabeled examples. In the spam detector, unlabeled examples are new emails that humans haven't yet labeled.

## Models

A model defines the relationship between feature(s) and label. For example, a span detector might associate certain features strongly with "spam". Let's highlight two phases of a model's life:

* __Training__ means __creating__ or __learning__ the model. That is, you show the model labeled examples and enable the model to gradually learn the relationships between features and label.

* __Inference__ means applying the trained model to unlabeled examples. That is, you use the trained model to make useful predictions (`y'`). For example, during __inference__, you can predict __MedianHouseValue__ for new unlabeled examples.

## Regression vs. classification

A __regression__ model predicts continuous values. For example, regression models make predictions that answer questions like the following:

* What is the value of a house in California?
* What is the probability that a user will click on this ad?

A __classification__ model predicts discrete values. For example, classification models make predictions that answer questions like the following:

* Is a given email message spam or not spam?
* Is this an image of a dog, a cat, or a hamster?

### Check your understanding

__Supervised Learning__

__Q1. Suppose you want to develop a supervised machine learning model to predict whether a given email is "spam" or "not spam." What are the true statements that you can think about for being a useful label?__

> * __Emails not marked as "spam" or "not spam" are unlabeled examples.__ <br>
>    Because our label consists of the values "spam" and "not spam", any email not yet marked as spam or not spam is an unlabeled example.
>
> * __The labels applied to some examples might be unreliable.__ <br>
>    Definitely. It's important to check how reliable your data is. The labels for this dataset probably come from email users who mark particular email messages as spam. Since most users do not mark every suspicious email message as spam, we may have trouble knowing whether an email is spam. Furthermore, spammers could intentionally poison our model by providing faulty labels.

__Features and Labels__

__Q1. Suppose an online shoe store wants to create a supervised ML model that will provide personalized shoe recommendations to users. That is, the model will recommend certain pairs of shoes to Marty and different pairs of shoes to Janet. The system will use past user behavior data to generate training data. What are the true statements that you can think of for being a useful label?__

> * __"Shoe size" is a useful feature.__ <br>
> "Shoe size" is a quantifiable signal that likely has a strong impact on whether the user will like the recommended shoes. For example, if Marty wears size 9, the model shouldn't recommend size 7 shoes.
>
> * __"The user clicked on the shoe's description" is a useful label.__ <br>
> Users probably only want to read more about those shoes that they like. Clicks by users is, therefore, an observable, quantifiable metric that could serve as a good training label. Since our training data derives from past user behavior, our labels need to derive from objective behaviors like clicks that strongly correlate with user preferences.


# Descending into ML

Linear regression is a method for finding the straight line or hyperplane that best fits a set of points.

## Linear Regression

It has long been known that crickets (an insect species) chirp more frequently on hotter days than on cooler days. For decades, professional and amateur scientists have cataloged data on chirps-per-minute and temperature. As a birthday gift, your Aunt Ruth gives you her cricket database and asks you to learn a model to predict this relationship. Using this data, you want to explore this relationship.

First, examine your data by plotting it:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/CricketPoints.svg' />

  <strong>Figure 1. Chirps per Minute vs. Temperature in Celsius.</strong>
</div>

As expected, the plot shows the temperature rising with the number of chirps. Is this relationship between chirps and temperature linear? Yes, you could draw a single straight line like the following to approximate this relationship:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/CricketLine.svg' />

  <strong>Figure 2. A linear relationship.</strong>
</div>

True, the line doesn't pass through every dot, but the line does clearly show the relationship between chirps and temperature. Using the equation for a line, you could write down this relationship as follows:

$$y=mx+b$$

where:

*  $y$ is the temperature in Celsius — the value we're trying to predict.
* $m$ is the slope of the line.
* $x$ is the number of chirps per minute — the value of our input feature.
* $b$ is the y-intercept.

By convention in machine learning, you'll write the equation for a model slightly differently:

$$y^\prime=b+w_{1}x_{1}$$

where:

* $y^\prime$ is a predicted [label](https://developers.google.com/machine-learning/crash-course/framing/ml-terminology#labels) (a desired output).
* $b$ is the bias (the y-intercept), sometimes referred to as $w_{0}$.
* $w_{1}$ is the [weight](https://developers.google.com/machine-learning/glossary#weight) of feature 1. Weight is the same concept as the "slope" _$m$_ in the traditional equation of a line.
* $x_{1}$ is a [feature](https://developers.google.com/machine-learning/crash-course/framing/ml-terminology#features) (a known input).

To __infer__ (predict) the temprature $y^\prime$ for new chirps-per-minute value $x_{1}$, just substitute the $x_{1}$ value into this model.

Although this model uses only one feature, a more sophisticated model might rely on multiple features, each having a separate weight ($w_{1}$, $w_{2}$, etc.). For example, a model that relies on three features might look as follows:

$$y^\prime=b+w_{1}x_{1}+w_{2}x_{2}+w_{3}x_{3}$$

## Training and Loss

__Training__ a model simply means learning (determining) good values for all the weights and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called __empirical risk minimization__.

Loss is the penalty for a bad prediction. That is, __loss__ is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples. For example, Figure 3 shows a high loss model on the left and a low loss model on the right. Note the following about the figure:

* The arrows represent loss.
* The blue lines represent predictions.

<div algin='center'>
  <img src='https://developers.google.com/machine-learning/crash-course/images/LossSideBySide.png' />

  <strong>Figure 3. High loss in the left model; low loss in the right model.</strong>
</div>

Notice that the arrows in the left plot are much longer than their counterparts in the right plot. Clearly, the line in the right plot is a much better predictive model than the line in the left plot.

You might be wondering whether you could create a mathematical function — a loss function — that would aggregate the individual losses in a meaningful fashion.

### Squared loss: a popular loss function

The linear regression models we'll examine here use a loss function called __squared loss__ (also known as __L2 loss__). The __squared loss__ for a single example is as follows:

```
  = the square of the difference between the label and the prediction
  = (observation - prediction(x))2
  = (y - y')2
```

__Mean square error (MSE)__ is the average squared loss per example over the whole dataset. To calculate __MSE__, sum up all the squared losses for individual examples and then divide by the number of examples:

$$\mathrm{MSE}=\dfrac{1}{N}\sum_{(x,y) \in D} (y - \mathrm{prediction}(x))^2$$

where:

* $(x,y)$ is an example in which
  * $x$ is a set of features (for example chirps/minute, age, gender) that the model uses to make predictions.
  * $y$ is the example's label (for example, temprature).
* $prediction(x)$ is a function of the weights and bias in combination with the sets of features $x$.
* $D$ is a dataset containing many labeled examples, which are $(x,y)$ pairs.
* $N$ is the number of examples in $D$.

Although MSE is commonly-used in machine learning, it is neither the only practical loss function nor the best loss function for all circumstances.

### Check your understanding

__Mean Squared Error__

Consider the following two plots:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/MCEDescendingIntoMLLeft.png' />

  <img src='https://developers.google.com/static/machine-learning/crash-course/images/MCEDescendingIntoMLRight.png' />
</div>

__Q1. Which of the two data sets shown in the preceding plots has the higher Mean Squared Error (MSE)?__

> __The dataset on the right.__ <br>
> The eight examples on the line incur a total loss of 0. However, although only two points lay off the line, both of those points are twice as far off the line as the outlier points in the left figure. Squared loss amplifies those differences, so an offset of two incurs a loss four times as great as an offset of one.
> $$\mathrm{MSE}=\dfrac{0^2+0^2+0^2+2^2+0^2+0^2+0^2+2^2+0^2+0^2}{10}=0.8$$

# Reducing Loss

To train a model, we need a good way to reduce the model’s loss. An iterative approach is one widely used method for reducing loss, and is as easy and efficient as walking down a hill.

## An iterative approach

Iterative learning might remind you of the "[Hot and Cold](http://www.howcast.com/videos/258352-how-to-play-hot-and-cold/)" kid's game for finding a hidden object like a thimble. In this game, the "hidden object" is the best possible model. You'll start with a wild guess ("The value of $w_{1}$ is $0$.") and wait for the system to tell you what the loss is. Then, you'll try another guess ("The value of $w_{1}$ is $0.5$.") and see what the loss is. Aah, you're getting warmer. Actually, if you play this game right, you'll usually be getting warmer. The real trick to the game is trying to find the best possible model as efficiently as possible.

The following figure suggests the iterative trial-and-error process that machine learning algorithms use to train a model:

<div align='center'>
  <img src="https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentDiagram.svg" />

  <strong>Figure 1. An iterative approach to training a model.</strong>
</div>

Iterative strategies are prevalent in machine learning, primarily because they scale so well to large data sets.

The "model" takes one or more features as input and returns one prediction ($y′$) as output. To simplify, consider a model that takes one feature and returns one prediction:

$$y′=b+w_{1}x_{1}$$

What initial values should we set for $b$ and $w_{1}$? For linear regression problems, it turns out that the starting values aren't important. We could pick random values, but we'll just take the following trivial values instead:

* $b=0$
* $w_{1}=0$

Suppose that the first feature value is 10. Plugging that feature value into the prediction function yields:

$$y′=0+0⋅10=0$$

The "Compute Loss" part of the diagram is the [loss function](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss) that the model will use. Suppose we use the squared loss function. The loss function takes in two input values:

* $y′$: The model's prediction for features $x$
* $y$: The correct label corresponding to features $x$

At last, we've reached the "Compute parameter updates" part of the diagram. It is here that the machine learning system examines the value of the loss function and generates new values for $b$ and $w_{1}$. For now, just assume that this mysterious box devises new values and then the machine learning system re-evaluates all those features against all those labels, yielding a new value for the loss function, which yields new parameter values. And the learning continues iterating until the algorithm discovers the model parameters with the lowest possible loss. Usually, you iterate until overall loss stops changing or at least changes extremely slowly. When that happens, we say that the model has [__converged__](https://developers.google.com/machine-learning/glossary#convergence).

## Gradient Descent

The iterative approach diagram contained a green hand-wavy box entitled "Compute parameter updates." We'll now replace that algorithmic fairy dust with something more substantial.

Suppose we had the time and the computing resources to calculate the loss for all possible values of $w_{1}$. For the kind of regression problems we've been examining, the resulting plot of loss vs. $w_{1}$ will always be convex. In other words, the plot will always be bowl-shaped, kind of like this:

<div align='center'>
  <img src="https://developers.google.com/static/machine-learning/crash-course/images/convex.svg" />

  <strong>Figure 2. Regression problems yield convex loss vs. weight plots.</strong>
</div>

Convex problems have only one minimum; that is, only one place where the slope is exactly 0. That minimum is where the loss function converges.

Calculating the loss function for every conceivable value of $w_{1}$ over the entire data set would be an inefficient way of finding the convergence point. Let's examine a better mechanism—very popular in machine learning—called __gradient descent__.

The first stage in gradient descent is to pick a starting value (a starting point) for $w_{1}$. The starting point doesn't matter much; therefore, many algorithms simply set $w_{1}$ to $0$ or pick a random value. The following figure shows that we've picked a starting point slightly greater than $0$:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentStartingPoint.svg' />
  
  <strong>Figure 3. A starting point for gradient descent.</strong>
</div>

The gradient descent algorithm then calculates the gradient of the loss curve at the starting point. Here in this Figure, the gradient of the loss is equal to the [derivative](https://wikipedia.org/wiki/Differential_calculus#The_derivative) (slope) of the curve, and tells you which way is "warmer" or "colder." When there are multiple weights, the gradient is a vector of partial derivatives with respect to the weights.

### Partial derivatives

A __multivariable function__ is a function with more than one argument, such as:

$$f(x,y)=e^{2y}\mathrm{sin}(x)$$

The __partial derivate__ $f$ __with respect to__ $x$, denoted as follows:

$$\dfrac{∂f}{∂x}$$

is a derivative of $f$ considered as a function of $x$ alone. To find the following:

$$\dfrac{∂f}{∂x}$$

so must hold $y$ constant (so $f$ is now a function of one variable $x$), and take the regular derivative of $f$ with respect to $x$. For example, when $y$ is fixed at $1$, the preceding function becomes:

$$f(x)=e^{2}\mathrm{sin}(x)$$

This is just a function of one variable $x$, whose derivative is:

$$e^{2}\mathrm{cos}(x)$$

In general, thinking of $y$ as fixed, the partial derivative of $f$ with respect to $x$ is calculated as follows:

$$\dfrac{∂f}{∂x}(x,y)=e^{2y}\mathrm{sin}(x)$$

Similarly, if we hold $x$ fixed instead, the partial derivative of $f$ with respect to $y$ is:

$$\dfrac{∂f}{∂x}(x,y)=2e^{2y}\mathrm{sin}(x)$$

Intuitively, a partial derivative tells how much the function changes when you perturb one variable a bit. In the preceding example:

$$\dfrac{\partial f}{\partial x}(0,1)=e^2 \approx 7.4$$

So when you start at $(0,1)$, hold $y$ constant, and move $x$ a little, $f$ changes by about $7.4$ times the amount that you changed $x$.

In machine learning, partial derivatives are mostly used in conjunction with the gradient of a function.

### Gradients

The __gradient__ of a function, denoted as follows, is the vector of partial derivatives with respect to all of the independent variables:

$$∇f$$

For instance, if:

$$f(x,y)=e^{2y}\mathrm{sin}(x)$$

then:

$$∇f(x,y)= \left( \dfrac{\partial f}{\partial x}(x,y), \dfrac{\partial f}{\partial y}(x,y) \right)=(e^{2y}\mathrm{cos}(x), 2e^{2y}\mathrm{sin}(x))$$

Note the following:

* $∇f$ - Points in the direction of greatest increase of the function.
* $−∇f$ - Points in the direction of greatest decrease of the function.

The number of dimensions in a vector is equal to the number of variables in the formula for $f$; in other words, the vector falls within the domain space of the function. For instance, the graph of the following function $f(x,y)$:

$$f(x,y)=4+(x−2)^2+2y^2$$

when viewed in three dimensions with $z=f(x,y)$ looks like a valley with a minimum at $(2,0,4)$:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/ThreeDimensionalPlot.svg' />
</div>

The gradient of $f(x,y)$ is a two-dimensional vector that tells you in which $(x,y)$ direction to move for the maximum increase in height. Thus, the negative of the gradient moves you in the direction of maximum decrease in height. In other words, the negative of the gradient vector points into the valley.

In machine learning, gradients are used in gradient descent. We often have a loss function of many variables that we are trying to minimize, and we try to do this by following the negative of the gradient of the function.

Okay, so comming back to our gradient descent. We've found that a gradient is a vector that has both of the following characterstics:

* A direction
* A magnitude

The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentNegativeGradient.svg' />

  <strong>Figure 4. Gradient descent relies on negative gradients.</strong>
</div>

To determine the next point along the loss function curve, the gradient descent algorithm adds some fraction of the gradient's magnitude to the starting point as shown in the following figure:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentGradientStep.svg' />

  <strong>Figure 5. A gradient step moves us to the next point on the loss curve.</strong>
</div>

The gradient descent then repeats this process, edging ever closer to the minimum.

## Learning Rate

As noted, the gradient vector has both a direction and a magnitude. Gradient descent algorithms multiply the gradient by a scalar known as the __learning rate__ (also sometimes called __step size__) to determine the next point. For example, if the gradient magnitude is $2.5$ and the learning rate is $0.01$, then the gradient descent algorithm will pick the next point $0.025$ away from the previous point.

__Hyperparameters__ are the knobs that programmers tweak in machine learning algorithms. Most machine learning programmers spend a fair amount of time tuning the learning rate. If you pick a learning rate that is too small, learning will take too long:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/LearningRateTooSmall.svg' />

  <strong>Figure 6. Learning rate is too small.</strong>
</div>

Conversely, if you specify a learning rate that is too large, the next point will perpetually bounce haphazardly across the bottom of the well like a quantum mechanics experiment gone horribly wrong:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/LearningRateTooLarge.svg' />

  <strong>Figure 7. Learning rate is too large.</strong>
</div>

There's a [Goldilocks](https://wikipedia.org/wiki/Goldilocks_principle) learning rate for every regression problem. The Goldilocks value is related to how flat the loss function is. If you know the gradient of the loss function is small then you can safely try a larger learning rate, which compensates for the small gradient and results in a larger step size.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/LearningRateJustRight.svg' />

  <strong>Figure 8. Learning rate is just right.</strong>
</div>

The ideal learning rate in one-dimension is $\dfrac{1}{f(x)″}$ (the inverse of the second derivative of $f(x)$ at $x$).

The ideal learning rate for $2$ or more dimensions is the inverse of the [Hessian](https://wikipedia.org/wiki/Hessian_matrix) (matrix of second partial derivatives).

The story for general convex functions is more complex.

## Stochastic Gradient Descent

__Stochastic Gradient Descent (SGD)__ is an optimization algorithm used to find the minimum of a function. It is a type of gradient descent algorithm that is often used in machine learning and deep learning. It is called "stochastic" because it uses random samples of the data to estimate the gradient of the objective function, rather than using the entire dataset.

Here's how it works:

* The algorithm starts with an initial guess of the parameters.
* It then selects a random sample of the data and calculates the gradient of the objective function with respect to the parameters using that sample.
* The parameters are then updated in the direction opposite to the gradient.
* This process is repeated until the parameters converge to a minimum of the objective function.

One of the main advantage of stochastic gradient descent is that it is computationally efficient. Because it uses a random sample of the data at each step, it can be much faster than batch gradient descent, which uses the entire dataset to calculate the gradient. Additionally, because it uses random samples, it can "escape" from local minima and converge to a global minimum.

An example of an application of __SGD__ is Linear Regression, where the objective function is the mean squared error and the parameters are the weights of the model. Another example is Logistic Regression, where the objective function is the cross-entropy loss and the parameters are the weights of the model.

It should be noted that the optimization speed of __SGD__ can be affected by the choice of the learning rate and the shuffling of the data during each iteration. Also it is not guaranteed to find the global minimum because of the randomness but it can be useful in practice.

In summary, __Stochastic Gradient Descent__ is an optimization algorithm that is efficient and can help to find the global minimum of a function. It has been widely used in machine learning and deep learning tasks.

### __Check Your Understanding__

__Q1. When performing gradient descent on a large data set, which of the following batch sizes will likely be more efficient?__

> __A small batch or even a batch of one example (SGD).__
>
> Amazingly enough, performing gradient descent on a small batch or even a batch of one example is usually more efficient than the full batch. After all, finding the gradient of one example is far cheaper than finding the gradient of millions of examples. To ensure a good representative sample, the algorithm scoops up another random small batch (or batch of one) on every iteration.

# Introduction to TensorFlow

TensorFlow is an end-to-end open source platform for machine learning. TensorFlow is a rich system for managing all aspects of a machine learning system; however, this class focuses on using a particular TensorFlow API to develop and train machine learning models. See the [TensorFlow documentation](https://tensorflow.org/) for complete details on the broader TensorFlow system.

TensorFlow APIs are arranged hierarchically, with the high-level APIs built on the low-level APIs. Machine learning researchers use the low-level APIs to create and explore new machine learning algorithms. In this class, you will use a high-level API named tf.keras to define and train machine learning models and to make predictions. tf.keras is the TensorFlow variant of the open-source [Keras](https://keras.io/) API.

The following figure shows the hierarchy of TensorFlow toolkits:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/TFHierarchyNew.svg' />

  <strong>Figure 1. TensorFlow toolkit hierarchy.</strong>
</div>

## Linear regression with tf.keras

## Simple Linear regression with Synthetic Data

```python
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
```

## Define functions that build and train a model

The following code defines two functions:

  * `build_model(my_learning_rate)`, which builds an empty model.
  * `train_model(model, feature, label, epochs)`, which trains the model from the examples (feature and label) you pass. 

Since you don't need to understand model building code right now, you may optionally explore this code.

```python
def build_model(learning_rate):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential. 
  # A sequential model contains one or more layers.
  model = tf.keras.models.Sequential()

  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer. 
  model.add(tf.keras.layers.Dense(units=1, 
                                  input_shape=(1,)))

  # Compile the model topography into code that 
  # TensorFlow can efficiently execute. Configure 
  # training to minimize the model's mean squared error. 
  model.compile(optimizer=tf.keras.optimizers.RMSprop(
                                    learning_rate=learning_rate),
                loss='mean_squared_error',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model           


def train_model(model, feature, label, epochs, batch_size):
  """Train the model by feeding it data."""

  # Feed the feature values and the label values to the 
  # model. The model will train for the specified number 
  # of epochs, gradually learning how the feature values
  # relate to the label values. 
  history = model.fit(x=feature,
                      y=label,
                      batch_size=batch_size,
                      epochs=epochs)

  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the 
  # rest of history.
  epochs = history.epoch
  
  # Gather the history (a snapshot) of each epoch.
  hist = pd.DataFrame(history.history)

  # Specifically gather the model's root mean 
  # squared error at each epoch. 
  rmse = hist['root_mean_squared_error']

  return trained_weight, trained_bias, epochs, rmse
```

## Define plotting functions

We're using a popular Python library called [Matplotlib](https://developers.google.com/machine-learning/glossary/#matplotlib) to create the following two plots:

*  a plot of the feature values vs. the label values, and a line showing the output of the trained model.
*  a [loss curve](https://developers.google.com/machine-learning/glossary/#loss_curve).

```python
def plot_the_model(trained_weight, trained_bias, feature, label):
  """Plot the trained model against the training feature and label."""

  # Label the axes.
  plt.xlabel('feature')
  plt.ylabel('label')

  # Plot the feature values vs. label values.
  plt.scatter(feature, label)

  # Create a red line representing the model. The red line starts
  # at coordinates (x0, y0) and ends at coordinates (x1, y1).
  x0 = 0
  y0 = trained_bias
  x1 = feature[-1]
  y1 = trained_bias + (trained_weight * x1)
  plt.plot([x0, x1], [y0, y1], c='r')

  # Render the scatter plot and the red line.
  plt.show()

def plot_the_loss_curve(epochs, rmse):
  """Plot the loss curve, which shows loss vs. epoch."""

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Root Mean Squared Error')

  plt.plot(epochs, rmse, label='Loss')
  plt.legend()
  plt.ylim([rmse.min() * 0.97, rmse.max()])
  plt.show()
```

## Define the dataset

The dataset consists of 12 [examples](https://developers.google.com/machine-learning/glossary/#example). Each example consists of one [feature](https://developers.google.com/machine-learning/glossary/#feature) and one [label](https://developers.google.com/machine-learning/glossary/#label).

```python
feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
label   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])
```

## Specify the hyperparameters

The hyperparameters in this Colab are as follows:

  * [learning rate](https://developers.google.com/machine-learning/glossary/#learning_rate)
  * [epochs](https://developers.google.com/machine-learning/glossary/#epoch)
  * [batch_size](https://developers.google.com/machine-learning/glossary/#batch_size)

The following code cell initializes these hyperparameters and then invokes the functions that build and train the model.

```python
learning_rate=0.01
epochs=10
batch_size=12

model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(model, feature, 
                                                         label, epochs,
                                                         batch_size)
plot_the_model(trained_weight, trained_bias, feature, label)
plot_the_loss_curve(epochs, rmse)
```

## Task 1: Examine the graphs

Examine the top graph. The blue dots identify the actual data; the red line identifies the output of the trained model. Ideally, the red line should align nicely with the blue dots.  Does it?  Probably not.

A certain amount of randomness plays into training a model, so you'll get somewhat different results every time you train.  That said, unless you are an extremely lucky person, the red line probably *doesn't* align nicely with the blue dots.  

Examine the bottom graph, which shows the loss curve. Notice that the loss curve decreases but doesn't flatten out, which is a sign that the model hasn't trained sufficiently.

## Task 2: Increase the number of epochs

Training loss should steadily decrease, steeply at first, and then more slowly. Eventually, training loss should eventually stay steady (zero slope or nearly zero slope), which indicates that training has [converged](http://developers.google.com/machine-learning/glossary/#convergence).

In Task 1, the training loss did not converge. One possible solution is to train for more epochs.  Your task is to increase the number of epochs sufficiently to get the model to converge. However, it is inefficient to train past convergence, so don't just set the number of epochs to an arbitrarily high value.

Examine the loss curve. Does the model converge?

```python
learning_rate=0.01
epochs=450
batch_size=12 

model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(model, feature, 
                                                         label, epochs,
                                                         batch_size)
plot_the_model(trained_weight, trained_bias, feature, label)
plot_the_loss_curve(epochs, rmse)
```

## Task 3: Increase the learning rate

In Task 2, you increased the number of epochs to get the model to converge. Sometimes, you can get the model to converge more quickly by increasing the learning rate. However, setting the learning rate too high often makes it impossible for a model to converge. In Task 3, we've intentionally set the learning rate too high. Run the following code cell and see what happens.

```python
learning_rate=100 
epochs=500
batch_size = batch_size 

model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(model, feature, 
                                                         label, epochs,
                                                         batch_size)
plot_the_model(trained_weight, trained_bias, feature, label)
plot_the_loss_curve(epochs, rmse)
```

The resulting model is terrible; the red line doesn't align with the blue dots. Furthermore, the loss curve oscillates like a [roller coaster](https://www.wikipedia.org/wiki/Roller_coaster).  An oscillating loss curve strongly suggests that the learning rate is too high. 

## Task 4: Find the ideal combination of epochs and learning rate

Assign values to the following two hyperparameters to make training converge as efficiently as possible: 

*  `learning_rate`
*  `epochs`

```python
learning_rate=0.14
epochs=70
batch_size = batch_size

model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(model, feature, 
                                                         label, epochs,
                                                         batch_size)
plot_the_model(trained_weight, trained_bias, feature, label)
plot_the_loss_curve(epochs, rmse)
```

## Task 5: Adjust the batch size

The system recalculates the model's loss value and adjusts the model's weights and bias after each **iteration**.  Each iteration is the span in which the system processes one batch. For example, if the **batch size** is 6, then the system recalculates the model's loss value and adjusts the model's weights and bias after processing every 6 examples.  

One **epoch** spans sufficient iterations to process every example in the dataset. For example, if the batch size is 12, then each epoch lasts one iteration. However, if the batch size is 6, then each epoch consumes two iterations.  

It is tempting to simply set the batch size to the number of examples in the dataset (12, in this case). However, the model might actually train faster on smaller batches. Conversely, very small batches might not contain enough information to help the model converge. 

Experiment with `batch_size` in the following code cell. What's the smallest integer you can set for `batch_size` and still have the model converge in a hundred epochs?

```python
learning_rate=0.05
epochs=125
batch_size=1 # Wow, a batch size of 1 works!

model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(model, feature, 
                                                         label, epochs,
                                                         batch_size)
plot_the_model(trained_weight, trained_bias, feature, label)
plot_the_loss_curve(epochs, rmse)
```

## Summary of hyperparameter tuning

Most machine learning problems require a lot of hyperparameter tuning.  Unfortunately, we can't provide concrete tuning rules for every model. Lowering the learning rate can help one model converge efficiently but make another model converge much too slowly.  You must experiment to find the best set of hyperparameters for your dataset. That said, here are a few rules of thumb:

 * Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero. 
 * If the training loss does not converge, train for more epochs.
 * If the training loss decreases too slowly, increase the learning rate. Note that setting the learning rate too high may also prevent training loss from converging.
 * If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
 * Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
 * Setting the batch size to a *very* small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
 * For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory. 

Remember: the ideal combination of hyperparameters is data dependent, so you must always experiment and verify.

# Linear Regression with a Real Dataset

Now we are going to use a real dataset to predict the prices of houses in California.

## The Dataset
  
The [dataset for this exercise](https://developers.google.com/machine-learning/crash-course/california-housing-data-description) is based on 1990 census data from California. The dataset is old but still provides a great opportunity to learn about machine learning programming.

```python
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
```

## The dataset

Datasets are often stored on disk or at a URL in [.csv format](https://wikipedia.org/wiki/Comma-separated_values). 

A well-formed .csv file contains column names in the first row, followed by many rows of data.  A comma divides each value in each row. For example, here are the first five rows of the .csv file holding the California Housing Dataset:

```
"longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"
-114.310000,34.190000,15.000000,5612.000000,1283.000000,1015.000000,472.000000,1.493600,66900.000000
-114.470000,34.400000,19.000000,7650.000000,1901.000000,1129.000000,463.000000,1.820000,80100.000000
-114.560000,33.690000,17.000000,720.000000,174.000000,333.000000,117.000000,1.650900,85700.000000
-114.570000,33.640000,14.000000,1501.000000,337.000000,515.000000,226.000000,3.191700,73400.000000
```

### Load the .csv file into a pandas DataFrame

Like many machine learning programs, we gather the `.csv` file and stores the data in memory as a pandas Dataframe. Pandas is an open source Python library. The primary datatype in pandas is a DataFrame.  You can imagine a pandas DataFrame as a spreadsheet in which each row is identified by a number and each column by a name. Pandas is itself built on another open source Python library called NumPy.

The following code cell imports the .csv file into a pandas DataFrame and scales the values in the label (`median_house_value`):

```python
# Import the dataset.
training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

# Scale the label.
training_df["median_house_value"] /= 1000.0

# Print the first rows of the pandas DataFrame.
training_df.head()
```

Scaling `median_house_value` puts the value of each house in units of thousands. Scaling will keep loss values and learning rates in a friendlier range.  

Although scaling a label is usually *not* essential, scaling features in a multi-feature model usually *is* essential.

## Examine the dataset

A large part of most machine learning projects is getting to know your data. The pandas API provides a `describe` function that outputs the following statistics about every column in the DataFrame:

* `count`, which is the number of rows in that column. Ideally, `count` contains the same value for every column. 

* `mean` and `std`, which contain the mean and standard deviation of the values in each column. 

* `min` and `max`, which contain the lowest and highest values in each column.

* `25%`, `50%`, `75%`, which contain various [quantiles](https://developers.google.com/machine-learning/glossary/#quantile).

```python
# Get statistics on the dataset.
training_df.describe()
```

### Task 1: Identify anomalies in the dataset

Do you see any anomalies (strange values) in the data?

> The maximum value (max) of several columns seems very high compared to the other quantiles. For example, example the total_rooms column. Given the quantile values (25%, 50%, and 75%), you might expect the max value of total_rooms to be approximately 5,000 or possibly 10,000. However, the max value is actually 37,937.
>
> When you see anomalies in a column, become more careful about using that column as a feature. That said, anomalies in potential features sometimes mirror anomalies in the label, which could make the column be (or seem to be) a powerful feature.

## Define functions that build and train a model

The following code defines two functions:

  * `build_model(my_learning_rate)`, which builds a randomly-initialized model.
  * `train_model(model, feature, label, epochs)`, which trains the model from the examples (feature and label) you pass. 

Since you don't need to understand model building code right now, you may optionally explore this code.

```python
def build_model(my_learning_rate):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  model.add(tf.keras.layers.Dense(units=1, 
                                  input_shape=(1,)))

  # Compile the model topography into code that TensorFlow can efficiently
  # execute. Configure training to minimize the model's mean squared error. 
  model.compile(optimizer=tf.keras.optimizers.RMSprop(
                              learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model        


def train_model(model, df, feature, label, epochs, batch_size):
  """Train the model by feeding it data."""

  # Feed the model the feature and the label.
  # The model will train for the specified number of epochs. 
  history = model.fit(x=df[feature],
                      y=df[label],
                      batch_size=batch_size,
                      epochs=epochs)

  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch
  
  # Isolate the error for each epoch.
  hist = pd.DataFrame(history.history)

  # To track the progression of training, we're going to take a snapshot
  # of the model's root mean squared error at each epoch. 
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse
```

## Define plotting functions

We're using a popular Python library called [Matplotlib](https://developers.google.com/machine-learning/glossary/#matplotlib) to create the following two plots:

*  a plot of the feature values vs. the label values, and a line showing the output of the trained model.
*  a [loss curve](https://developers.google.com/machine-learning/glossary/#loss_curve).

```python
def plot_the_model(trained_weight, trained_bias, feature, label):
  """Plot the trained model against 200 random training examples."""

  # Label the axes.
  plt.xlabel(feature)
  plt.ylabel(label)

  # Create a scatter plot from 200 random points of the dataset.
  random_examples = training_df.sample(n=200)
  plt.scatter(random_examples[feature], random_examples[label])

  # Create a red line representing the model. The red line starts
  # at coordinates (x0, y0) and ends at coordinates (x1, y1).
  x0 = 0
  y0 = trained_bias
  x1 = random_examples[feature].max()
  y1 = trained_bias + (trained_weight * x1)
  plt.plot([x0, x1], [y0, y1], c='r')

  # Render the scatter plot and the red line.
  plt.show()


def plot_the_loss_curve(epochs, rmse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min() * 0.97, rmse.max()])
  plt.show()
```

## Call the model functions

An important part of machine learning is determining which [features](https://developers.google.com/machine-learning/glossary/#feature) correlate with the [label](https://developers.google.com/machine-learning/glossary/#label). For example, real-life home-value prediction models typically rely on hundreds of features and synthetic features. However, this model relies on only one feature. For now, you'll arbitrarily use `total_rooms` as that feature. 


```python
# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 30
batch_size = 30

# Specify the feature and the label.
feature = "total_rooms"  # the total number of rooms on a specific city block.
label="median_house_value" # the median value of a house on a specific city block.
# That is, you're going to create a model that predicts house value based 
# solely on total_rooms.  

# Discard any pre-existing version of the model.
model = None

# Invoke the functions.
model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(model, training_df, 
                                         feature, label,
                                         epochs, batch_size)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias )

plot_the_model(weight, bias, feature, label)
plot_the_loss_curve(epochs, rmse)
```

A certain amount of randomness plays into training a model. Consequently, you'll get different results each time you train the model. That said, given the dataset and the hyperparameters, the trained model will generally do a poor job describing the feature's relation to the label.

## Use the model to make predictions

You can use the trained model to make predictions. In practice, [you should make predictions on examples that are not used in training](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data). However, for this exercise, you'll just work with a subset of the same training dataset.

First, run the following code to define the house prediction function:

```python
def predict_house_values(n, feature, label):
  """Predict house values based on a feature."""

  batch = training_df[feature][10000:10000 + n]
  predicted_values = model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand$   in thousand$")
  print("--------------------------------------")
  for i in range(n):
    print ("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
                                   training_df[label][10000 + i],
                                   predicted_values[i][0] ))
```

Now, invoke the house prediction function on 10 examples:

```python
predict_house_values(10, feature, label)
```

### Task 2: Judge the predictive power of the model

Look at the preceding table. How close is the predicted value to the label value?  In other words, does your model accurately predict house values?

> Most of the predicted values differ significantly from the label value, so the trained model probably doesn't have much predictive power. However, the first 10 examples might not be representative of the rest of the examples.  

## Task 3: Try a different feature

The `total_rooms` feature had only a little predictive power. Would a different feature have greater predictive power?  Try using `population` as the feature instead of `total_rooms`. 

Note: When you change features, you might also need to change the hyperparameters.

```python
# Pick a feature other than "total_rooms"
feature = "population"

# Possibly, experiment with the hyperparameters.
learning_rate = 0.05
epochs = 18
batch_size = 3

# Don't change anything below.
model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(model, training_df, 
                                         feature, label,
                                         epochs, batch_size)

plot_the_model(weight, bias, feature, label)
plot_the_loss_curve(epochs, rmse)

predict_house_values(10, feature, label)
```

Did `population` produce better predictions than `total_rooms`?

> Training is not entirely deterministic, but population typically converges at a slightly higher RMSE than total_rooms. So, population appears to be about the same or slightly worse at making predictions than total_rooms.

## Task 4: Define a synthetic feature

You have determined that `total_rooms` and `population` were not useful features.  That is, neither the total number of rooms in a neighborhood nor the neighborhood's population successfully predicted the median house price of that neighborhood. Perhaps though, the *ratio* of `total_rooms` to `population` might have some predictive power. That is, perhaps block density relates to median house value.

To explore this hypothesis, do the following: 

1. Create a [synthetic feature](https://developers.google.com/machine-learning/glossary/#synthetic_feature) that's a ratio of `total_rooms` to `population`.
2. Tune the three hyperparameters.
3. Determine whether this synthetic feature produces 
   a lower loss value than any of the single features you 
   tried earlier.

```python
# Define a synthetic feature
training_df["rooms_per_person"] = training_df["total_rooms"] / training_df["population"]
feature = "rooms_per_person"

# Tune the hyperparameters.
learning_rate = 0.06
epochs = 24
batch_size = 30

# Don't change anything below this line.
model = build_model(learning_rate)
weight, bias, epochs, mae = train_model(model, training_df,
                                        feature, label,
                                        epochs, batch_size)

plot_the_model(weight, bias, feature, label)
plot_the_loss_curve(epochs, mae)
predict_house_values(15, feature, label)
```

Based on the loss values, this synthetic feature produces a better model than the individual features you tried in Task 2 and Task 3. However, the model still isn't creating great predictions.

## Task 5. Find feature(s) whose raw values correlate with the label

So far, we've relied on trial-and-error to identify possible features for the model.  Let's rely on statistics instead.

A **correlation matrix** indicates how each attribute's raw values relate to the other attributes' raw values. Correlation values have the following meanings:

  * `1.0`: perfect positive correlation; that is, when one attribute rises, the other attribute rises.
  * `-1.0`: perfect negative correlation; that is, when one attribute rises, the other attribute falls. 
  * `0.0`: no correlation; the two columns [are not linearly related](https://en.wikipedia.org/wiki/Correlation_and_dependence#/media/File:Correlation_examples2.svg).

In general, the higher the absolute value of a correlation value, the greater its predictive power. For example, a correlation value of -0.8 implies far more predictive power than a correlation of -0.2.

The following code cell generates the correlation matrix for attributes of the California Housing Dataset:

```python
# Generate a correlation matrix.
training_df.corr()
```

The correlation matrix shows nine potential features (including a synthetic
feature) and one label (`median_house_value`).  A strong negative correlation or strong positive correlation with the label suggests a potentially good feature.  

**Your Task:** Determine which of the nine potential features appears to be the best candidate for a feature?

> The median_income correlates 0.7 with the label (median_house_value), so median_income might be a good feature. The other seven potential features all have a correlation relatively close to 0.

```python
feature = "median_income"

# Possibly, experiment with the hyperparameters.
learning_rate = 0.01
epochs = 10
batch_size = 3

# Don't change anything below.
model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(model, training_df, 
                                         feature, label,
                                         epochs, batch_size)

plot_the_model(weight, bias, feature, label)
plot_the_loss_curve(epochs, rmse)

predict_house_values(10, feature, label)
```