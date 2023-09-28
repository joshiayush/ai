# Introduction to ML

Machine learning is a field of inquiry devoted to understanding and building methods that __'learn'__, that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence.

## Additional information

* _Rules of Machine Learning,_ [Rule #1: Don't be afraid to launch a product without machine learning](https://developers.google.com/machine-learning/rules-of-ml/#rule_1_dont_be_afraid_to_launch_a_product_without_machine_learning)

## Framing

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


