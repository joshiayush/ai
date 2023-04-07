# Representation

A machine learning model can't directly see, hear, or sense input examples. Instead, you must create a representation of the data to provide the model with a useful vantage point into the data's key qualities. That is, in order to train a model, you must choose the set of features that best represent the data.

## Feature Engineering

In traditional programming, the focus is on code. In machine learning projects, the focus shifts to representation. That is, one way developers hone a model is by adding and improving its features.

## Mapping Raw Data to Features

The left side of Figure 1 illustrates raw data from an input data source; the right side illustrates a __feature vector__, which is the set of floating-point values comprising the examples in your data set. __Feature engineering__ means transforming raw data into a feature vector. Expect to spend significant time doing feature engineering.

Many machine learning models must represent the features as real-numbered vectors since the feature values must be multiplied by the model weights.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/RawDataToFeatureVector.svg' />

  <strong>Figure 1. Feature engineering maps raw data to ML features.</strong>
</div>

### Mapping numeric values

Integer and floating-point data don't need a special encoding because they can be multiplied by a numeric weight. As suggested in Figure 2, converting the raw integer value 6 to the feature value 6.0 is trivial:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/FloatingPointFeatures.svg' />

  <strong>Figure 2. Mapping integer values to floating-point values.</strong>
</div>

### Mapping categorical values

[Categorical features](https://developers.google.com/machine-learning/glossary#categorical_data) have a discrete set of possible values. For example, there might be a feature called `street_name` with options that include:

```python
{'Charleston Road', 'North Shoreline Boulevard', 'Shorebird Way', 'Rengstorff Avenue'}
```

Since models cannot multiply strings by the learned weights, we use feature engineering to convert strings to numeric values.

We can accomplish this by defining a mapping from the feature values, which we'll refer to as the __vocabulary__ of possible values, to integers. Since not every street in the world will appear in our dataset, we can group all other streets into a catch-all "other" category, known as an __OOV (out-of-vocabulary) bucket__.

Using this approach, here's how we can map our street names to numbers:

* map Charleston Road to 0
* map North Shoreline Boulevard to 1
* map Shorebird Way to 2
* map Rengstorff Avenue to 3
* map everything else (OOV) to 4

However, if we incorporate these index numbers directly into our model, it will impose some constraints that might be problematic:

* We'll be learning a single weight that applies to all streets. For example, if we learn a weight of 6 for `street_name`, then we will multiply it by 0 for Charleston Road, by 1 for North Shoreline Boulevard, 2 for Shorebird Way and so on. Consider a model that predicts house prices using `street_name` as a feature. It is unlikely that there is a linear adjustment of price based on the street name, and furthermore this would assume you have ordered the streets based on their average house price. Our model needs the flexibility of learning different weights for each street that will be added to the price estimated using the other features.
* We aren't accounting for cases where `street_name` may take multiple values. For example, many houses are located at the corner of two streets, and there's no way to encode that information in the `street_name` value if it contains a single index.

To remove both these constraints, we can instead create a binary vector for each categorical feature in our model that represents values as follows:

* For values that apply to the example, set corresponding vector elements to `1`.
* Set all other elements to `0`.

The length of this vector is equal to the number of elements in the vocabulary. This representation is called a __one-hot__ encoding when a single value is 1, and a __multi-hot__ encoding when multiple values are 1.

Figure 3 illustrates a one-hot encoding of a particular street: Shorebird Way. The element in the binary vector for Shorebird Way has a value of `1`, while the elements for all other streets have values of `0`.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/OneHotEncoding.svg' />

  <strong>Figure 3. Mapping street address via one-hot encoding.</strong>
</div>

This approach effectively creates a Boolean variable for every feature value (e.g., street name). Here, if a house is on Shorebird Way then the binary value is 1 only for Shorebird Way. Thus, the model uses only the weight for Shorebird Way.

Similarly, if a house is at the corner of two streets, then two binary values are set to 1, and the model uses both their respective weights.

## Sparse Representation

Suppose that you had 1,000,000 different street names in your data set that you wanted to include as values for `street_name`. Explicitly creating a binary vector of 1,000,000 elements where only 1 or 2 elements are true is a very inefficient representation in terms of both storage and computation time when processing these vectors. In this situation, a common approach is to use a [sparse representation](https://developers.google.com/machine-learning/glossary#sparse_representation) in which only nonzero values are stored. In sparse representations, an independent model weight is still learned for each feature value, as described above.

## Qualities of Good Features

We've explored ways to map raw data into suitable feature vectors, but that's only part of the work. We must now explore what kinds of values actually make good features within those feature vectors.

## Avoid rarely used discrete feature values

Good feature values should appear more than 5 or so times in a data set. Doing so enables a model to learn how this feature value relates to the label. That is, having many examples with the same discrete value gives the model a chance to see the feature in different settings, and in turn, determine when it's a good predictor for the label. For example, a `house_type` feature would likely contain many examples in which its value was `victorian`:

```python
house_type: 'victorian'
```

Conversely, if a feature's value appears only once or very rarely, the model can't make predictions based on that feature. For example, `unique_house_id` is a bad feature because each value would be used only once, so the model couldn't learn anything from it:

```python
unique_house_id: '8SK982ZZ1242Z'
```

## Prefer clear and obvious meanings

Each feature should have a clear and obvious meaning to anyone on the project. For example, the following good feature is clearly named and the value makes sense with respect to the name:

```python
house_age_years: 27
```

Conversely, the meaning of the following feature value is pretty much indecipherable to anyone but the engineer who created it:

```python
house_age: 851472000
```

In some cases, noisy data (rather than bad engineering choices) causes unclear values. For example, the following user_age_years came from a source that didn't check for appropriate values:

```python
user_age_years: 277
```

## Don't mix "magic" values with actual data

Good floating-point features don't contain peculiar out-of-range discontinuities or "magic" values. For example, suppose a feature holds a floating-point value between 0 and 1. So, values like the following are fine:

```python
quality_rating: 0.82
quality_rating: 0.37
```

However, if a user didn't enter a `quality_rating`, perhaps the data set represented its absence with a magic value like the following:

```python
quality_rating: -1
```

To explicitly mark magic values, create a Boolean feature that indicates whether or not a `quality_rating` was supplied. Give this Boolean feature a name like `is_quality_rating_defined`.

In the original feature, replace the magic values as follows:

* For variables that take a finite set of values (discrete variables), add a new value to the set and use it to signify that the feature value is missing.
* For continuous variables, ensure missing values do not affect the model by using the mean value of the feature's data.

## Account for upstream instability

The definition of a feature shouldn't change over time. For example, the following value is useful because the city name _probably_ won't change. (Note that we'll still need to convert a string like "br/sao_paulo" to a one-hot vector.)

```python
city_id: 'br/sao_paulo'
```

But gathering a value inferred by another model carries additional costs. Perhaps the value "219" currently represents Sao Paulo, but that representation could easily change on a future run of the other model:

```python
inferred_city_cluster: 219
```