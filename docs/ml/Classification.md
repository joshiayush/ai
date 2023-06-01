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