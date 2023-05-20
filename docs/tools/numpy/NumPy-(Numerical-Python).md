# NumPy (Numerical Python)

_No matter what the data are, the first step in making them analyzable will be to transform them into arrays of numbers._

NumPy arrays form the core of nearly the entire ecosystem of data science tools in Python, so time spent learning to use NumPy effectively will be valuable no matter what aspect of data science interests you.

```python
import numpy as np
np.__version__
```

### Output


```
'1.24.0'
```

NumPy-style arrays are totally different from Python-style arrays, where Python style arrays provide this flexibility of being heterogenuous over the cost of efficiency NumPy does not. Fixed type NumPy-style arrays may lack in flexibility, but are much more efficient for storing and manipulating data.

<div align='center'>
  <img src='../media/numpy/numpy_arrays_vs_python_arrays.png' width='1000' height='600' />
</div>

## Creating NumPy Arrays

We'll start with the standard NumPy import, under the alias `np`:

```python
import numpy as np
```

### Creating arrays from Python lists

```python
np.array([1, 2, 3, 4, 5])
```

### Output


```
array([1, 2, 3, 4, 5])
```

Remember that unlike Python lists, NumPy is constrained to arrays that all contain the same type. If types do not match, NumPy will upcast if possible (here, integers are upcast to floating point):

```python
np.array([3.14, 5, 2, 1])
```

### Output


```
array([3.14, 5.  , 2.  , 1.  ])
```

If we want to explicitly set the data type of the resulting array, we can use the `dtype` keyword:

```python
np.array([3, 5, 2, 1], dtype='float32')
```

### Output


```
array([3., 5., 2., 1.], dtype=float32)
```

Finally, unlike Python lists, NumPy arrays can explicitly be multidimensional; here's one way of initializing a multidimensional array using a list of lists:

```python
np.array([range(i, i + 3) for i in [2, 4, 6]])
```

### Output


```
array([[2, 3, 4],
       [4, 5, 6],
       [6, 7, 8]])
```

The inner lists are treated as rows of the resulting two-dimensional array.

### Creating arrays from scratch

Especially for larger arrays, it is more efficient to create arrays from scratch using routines built into NumPy.

```python
# Create a length-10 integer array filled with zeros.
np.zeros(10, dtype=int)
```

### Output


```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

```python
# Create a 3x5 floating-point array filled with 1s.
np.ones((3, 5), dtype=float)
```

### Output


```
array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])
```

```python
# Create a 3x5 floating-point array filled with 3.14
np.full((3, 5), 3.14, dtype=float)
```

### Output


```
array([[3.14, 3.14, 3.14, 3.14, 3.14],
       [3.14, 3.14, 3.14, 3.14, 3.14],
       [3.14, 3.14, 3.14, 3.14, 3.14]])
```

```python
# Create an array filled with a linear sequence
# Starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
np.arange(0, 20, 2)
```

### Output


```
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
```

```python
# Create an array of five values evenly spaced between 0 and 1.
np.linspace(0, 1, 5)
```

### Output


```
array([0.  , 0.25, 0.5 , 0.75, 1.  ])
```

```python
# Create a 3x3 array of uniformly distributed random
# values between 0 and 1
np.random.random((3, 3))
```

### Output


```
array([[0.40276416, 0.4247601 , 0.66838607],
       [0.54128026, 0.75879512, 0.09090431],
       [0.93579692, 0.66825775, 0.35487241]])
```

```python
# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))
```

### Output


```
array([[ 0.55069101, -0.75163452, -0.17419984],
       [ 0.93847916,  1.13059429,  1.45485808],
       [ 0.54093265, -0.74499343,  0.18688181]])
```

```python
# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))
```

### Output


```
array([[3, 9, 8],
       [7, 9, 6],
       [7, 0, 1]])
```

```python
# Create a 3x3 identity matrix
np.eye(3)
```

### Output


```
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```

```python
# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that
# memory location
np.empty(3)
```

### Output


```
array([1., 1., 1.])
```