# Image classification

This short introduction uses [Keras](https://www.tensorflow.org/guide/keras/overview) to:

1. Load a prebuilt dataset.
2. Build a neural network machine learning model that classifies images.
3. Train this neural network.
4. Evaluate the accuracy of the model.

## Set up Tensorflow

Import `tensorflow` in your program to get started:

```python
import tensorflow as tf

print('Tensorflow version: ', tf.__version__)
```

## Load a dataset

Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The pixel values of the images range from 0 through 255. Scale these values to a range of 0 to 1 by dividing the values by `255.0`. This also converts the sample data from integers to floating-point numbers:

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

## Build a machine learning model

Build a [`tf.keras.Sequential`](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) model:

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

[Sequential](https://www.tensorflow.org/guide/keras/sequential_model) is useful for stacking layers where each layer has one input [tensor](https://www.tensorflow.org/guide/tensor) and one output tensor. Layers are functions with a known mathematical structure that can be reused and have trainable variables. Most TensorFlow models are composed of layers. This model uses the [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten), [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense), and [Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) layers.

For each example, the model returns a vector of [logits](https://developers.google.com/machine-learning/glossary#logitshttps://developers.google.com/machine-learning/glossary#logits) or [log-odds](https://developers.google.com/machine-learning/glossary#log-odds) scores, one for each class.

```python
predictions = model(x_train[:1]).numpy()
predictions
```

The [`tf.nn.softmax`](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) function converts these logits to probabilities for each class:

```python
tf.nn.softmax(predictions).numpy()
```

Define a loss function for training using [`losses.SparseCategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy):

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

The loss function takes a vector of ground truth values and a vector of logits and returns a scalar loss for each example. This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.

This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to `-tf.math.log(1/10) ~= 2.3`.

```python
loss_fn(y_train[:1], predictions).numpy()
```

Before you start training, configure and compile the model using Keras [`Model.compile`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile). Set the optimizer class to `adam`, set the `loss` to the `loss_fn` function you defined earlier, and specify a metric to be evaluated for the model by setting the `metrics` parameter to `accuracy`.

```python
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

## Train and evaluate your model

Use the [`Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) method to adjust your model parameters and minimize the loss:

```python
model.fit(x_train, y_train, epochs=5)
```

The [`Model.evaluate`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate) method checks the model's performance, usually on a [validation set](https://developers.google.com/machine-learning/glossary#validation-set) or [test set](https://developers.google.com/machine-learning/glossary#test-set).

```python
model.evaluate(x_test,  y_test, verbose=2)
```

The image classifier is now trained to ~98% accuracy on this dataset.

If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:

```python
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])
```