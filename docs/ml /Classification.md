# Classification

## Thresholding

Logistic regression returns a probability. You can use the returned probability "as is" (for example, the probability that the user will click on this ad is 0.00023) or convert the returned probability to a binary value (for example, this email is spam).

A logistic regression model that returns 0.9995 for a particular email message is predicting that it is very likely to be spam. Conversely, another email message with a prediction score of 0.0003 on that same logistic regression model is very likely not spam. However, what about an email message with a prediction score of 0.6? In order to map a logistic regression value to a binary category, you must define a __classification threshold__ (also called the __decision threshold__). A value above that threshold indicates "spam"; a value below indicates "not spam." It is tempting to assume that the classification threshold should always be 0.5, but thresholds are problem-dependent, and are therefore values that you must tune.

## True vs. False and Positive vs. Negative

In this section, we'll define the primary building blocks of the metrics we'll use to evaluate classification models. But first, a fable:

> __An Aesop's Fable: The Boy Who Cried Wolf (compressed)__
>
> A shepherd boy gets bored tending the town's flock. To have some fun, he cries out, "Wolf!" even though no wolf is in sight. The villagers run to protect the flock, but then get really mad when they realize the boy was playing a joke on them.
>
> [Iterate previous paragraph N times.]
>
> One night, the shepherd boy sees a real wolf approaching the flock and calls out, "Wolf!" The villagers refuse to be fooled again and stay in their houses. The hungry wolf turns the flock into lamb chops. The town goes hungry. Panic ensues.

Let's make the following definitions:

* "Wolf" is a __positive class__.
* "No wolf" is a __negative class__.

We can summarize our "wolf-prediction" model using a 2x2 [confusion matrix](https://developers.google.com/machine-learning/glossary#confusion_matrix) that depicts all four possible outcomes:

* __True Positive (TP):__
  * Reality: A wolf threatened.
  * Shepherd said: "Wolf."
  * Outcome: Shepherd is a hero.

* __False Positive (FP):__
  * Reality: No wolf threatened.
  * Shepherd said: "Wolf."
  * Outcome: Villagers are angry at shepherd for waking them up.

* __True Negative (TN):__
  * Reality: No wolf threatened.
  * Shepherd said: "No wolf."
  * Outcome: Everyone is fine.

* __False Negative (FN):__
  * Reality: A wolf threatened.
  * Shepherd said: "No wolf."
  * Outcome: The wolf ate all the sheep.

A __true positive__ is an outcome where the model correctly predicts the positive class. Similarly, a __true negative__ is an outcome where the model correctly predicts the negative class.

A __false positive__ is an outcome where the model incorrectly predicts the positive class. And a __false negative__ is an outcome where the model incorrectly predicts the negative class.

## Accuracy

Accuracy is one metric for evaluating classification models. Informally, __accuracy__ is the fraction of predictions our model got right. Formally, accuracy has the following definition:

$\mathrm{Accuracy} = \dfrac{\mathrm{Number\ of\ correct\ predictions}}{\mathrm{Total\ number\ of\ predictions}}$

For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:

$\mathrm{Accuracy} = \dfrac{TP + TN}{TP + TN + FP + FN}$

Where _TP_ = True Positives, _TN_ = True Negatives, _FP_ = False Positives, and _FN_ = False Negatives.

Let's try calculating accuracy for the following model that classified 100 tumors as [malignant](https://wikipedia.org/wiki/Malignancy) (the positive class) or [benign](https://wikipedia.org/wiki/Benign_tumor) (the negative class):

* __True Positive (TP):__
  * Reality: Malignant
  * ML model predicted: Malignant
  * __Number of TP results__: 1

* __False Positive (FP):__
  * Reality: Benign
  * ML model predicted: Malignant
  * __Number of TP results__: 1

* __True Negative (TN):__
  * Reality: Benign
  * ML model predicted: Benign
  * __Number of TP results__: 90

* __False Negative (FN):__
  * Reality: Malignant
  * ML model predicted: Benign
  * __Number of TP results__: 8

$\mathrm{Accuracy} = \dfrac{TP + TN}{TP + TN + FP + FN} = \dfrac{1 + 90}{1 + 90 + 1 + 8} = 0.91$

Accuracy comes out to 0.91, or 91% (91 correct predictions out of 100 total examples). That means our tumor classifier is doing a great job of identifying malignancies, right?

Actually, let's do a closer analysis of positives and negatives to gain more insight into our model's performance.

Of the 100 tumor examples, 91 are benign (90 TNs and 1 FP) and 9 are malignant (1 TP and 8 FNs).

Of the 91 benign tumors, the model correctly identifies 90 as benign. That's good. However, of the 9 malignant tumors, the model only correctly identifies 1 as malignant—a terrible outcome, as 8 out of 9 malignancies go undiagnosed!

While 91% accuracy may seem good at first glance, another tumor-classifier model that always predicts benign would achieve the exact same accuracy (91/100 correct predictions) on our examples. In other words, our model is no better than one that has zero predictive ability to distinguish malignant tumors from benign tumors.

Accuracy alone doesn't tell the full story when you're working with a __class-imbalanced data set__, like this one, where there is a significant disparity between the number of positive and negative labels.