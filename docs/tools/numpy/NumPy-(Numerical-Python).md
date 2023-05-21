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
array([[0.65279032, 0.63505887, 0.99529957],
       [0.58185033, 0.41436859, 0.4746975 ],
       [0.6235101 , 0.33800761, 0.67475232]])
```

```python
# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))
```

### Output


```
array([[ 1.0657892 , -0.69993739,  0.14407911],
       [ 0.3985421 ,  0.02686925,  1.05583713],
       [-0.07318342, -0.66572066, -0.04411241]])
```

```python
# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))
```

### Output


```
array([[7, 2, 9],
       [2, 3, 3],
       [2, 3, 4]])
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

## NumPy standard data types

NumPy arrays contain values of a single type, so it is important to have detailed knowledge of those types and their limitations. Because NumPy is built in C, the types will be familiar to users to C.

The standard NumPy data types are listed in the following table. Note that when constructing an array, you can specify them using a string:

```python
np.zeros(10, dtype='int16')
```

### Output


```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int16)
```

Or using the associated NumPy object:

```python
np.zeros(10, dtype=np.int16)
```

### Output


```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int16)
```

|  Data Type | Description                                                                   |
| ---------: | :---------------------------------------------------------------------------- |
|     bool\_ | Boolean (True or False) stored as byte                                        |
|      int\_ | Default integer type (same as C long; normally either int64 or int32)         |
|       intc | Identical to C int (normally int32 or int64)                                  |
|       intp | Integer used for indexing (same as C ssize_t; normally either int32 or int64) |
|       int8 | Byte (-128 to 127)                                                            |
|      int16 | Integer (-32768 to 32767)                                                     |
|      int32 | Integer (-2147483648 to 2147483648)                                           |
|      int64 | Integer (-9223372036854775808 to 9223372036854775807)                         |
|      uint8 | Unsigned integer (0 to 255)                                                   |
|     uint16 | Unsigned integer (0 to 65535)                                                 |
|     uint32 | Unsigned integer (0 to 4294967295)                                            |
|     uint64 | Unsigned integer (0 to 18446744073709551615)                                  |
|    float\_ | Shorthand for float64                                                         |
|    float16 | Half-precision float; sign bit, 5 bits exponent, 10 bits mantisa              |
|    float32 | Single-precision float; sign bit, 8 bits exponent, 23 bits mantisa            |
|    float64 | Double-precision float; sign bit, 11 bits exponent, 52 bits mantisa           |
|  complex\_ | Shorthand for complex128                                                      |
|  complex64 | Complex number, represented by two 32-bit floats                              |
| complex128 | Complex number, represented by two 64-bit floats                              |

## The Basics of NumPy Arrays

### NumPy Array Attributes

We'll start by defining three random arrays: a one-dimensional, two-dimensional, three-dimensional array. We’ll use NumPy’s random number generator, which we will _seed_ with a set value in order to ensure that the same random arrays are generated each time this code is run:

```python
import numpy as np
np.random.seed(0)

x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3, 4))
x3 = np.random.randint(10, size=(3, 4, 5))
```

Each array has attributes `ndim` (the number of dimensions), `shape` (the size of each dimension), and `size` (the total size of the array):

```python
print('x3 ndim:', x3.ndim)
print('x3 shape:', x3.shape)
print('x3 size:', x3.size)
```

### Output


```
x3 ndim: 3
x3 shape: (3, 4, 5)
x3 size: 60
```

Another useful attribute is the `dtype`, the data type of the array:

```python
print('x3 dtype:', x3.dtype)
```

### Output


```
x3 dtype: int64
```

Other attributes include `itemsize`, which lists the size (in bytes) of each array element, and `nbytes`, which lists the total size (in bytes) of the array:

```python
print('itemsize:', x3.itemsize, 'bytes')
print('nbytes:', x3.nbytes, 'bytes')
```

### Output


```
itemsize: 8 bytes
nbytes: 480 bytes
```

In general, we expect that `nbytes` is equal to `itemsize` times `size`.

```python
def np_array_generalization_check(np_array: np.ndarray) -> bool:
  if np_array.nbytes == np_array.itemsize * np_array.size:
    return True
  return False
```

```python
print('Generalization check result for x1:', np_array_generalization_check(x1))
print('Generalization check result for x2:', np_array_generalization_check(x2))
print('Generalization check result for x3:', np_array_generalization_check(x3))
```

### Output


```
Generalization check result for x1: True
Generalization check result for x2: True
Generalization check result for x3: True
```