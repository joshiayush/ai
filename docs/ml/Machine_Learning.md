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