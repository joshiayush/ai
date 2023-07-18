# NumPy (Numerical Python)

_No matter what the data are, the first step in making them analyzable will be to transform them into arrays of numbers._

NumPy arrays form the core of nearly the entire ecosystem of data science tools in Python, so time spent learning to use NumPy effectively will be valuable no matter what aspect of data science interests you.

```python
import numpy as np
np.__version__
```

###### Output


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

###### Output


```
array([1, 2, 3, 4, 5])
```

Remember that unlike Python lists, NumPy is constrained to arrays that all contain the same type. If types do not match, NumPy will upcast if possible (here, integers are upcast to floating point):

```python
np.array([3.14, 5, 2, 1])
```

###### Output


```
array([3.14, 5.  , 2.  , 1.  ])
```

If we want to explicitly set the data type of the resulting array, we can use the `dtype` keyword:

```python
np.array([3, 5, 2, 1], dtype='float32')
```

###### Output


```
array([3., 5., 2., 1.], dtype=float32)
```

Finally, unlike Python lists, NumPy arrays can explicitly be multidimensional; here's one way of initializing a multidimensional array using a list of lists:

```python
np.array([range(i, i + 3) for i in [2, 4, 6]])
```

###### Output


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

###### Output


```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

```python
# Create a 3x5 floating-point array filled with 1s.
np.ones((3, 5), dtype=float)
```

###### Output


```
array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])
```

```python
# Create a 3x5 floating-point array filled with 3.14
np.full((3, 5), 3.14, dtype=float)
```

###### Output


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

###### Output


```
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
```

```python
# Create an array of five values evenly spaced between 0 and 1.
np.linspace(0, 1, 5)
```

###### Output


```
array([0.  , 0.25, 0.5 , 0.75, 1.  ])
```

```python
# Create a 3x3 array of uniformly distributed random
# values between 0 and 1
np.random.random((3, 3))
```

###### Output


```
array([[0.62432256, 0.401882  , 0.28524826],
       [0.84660473, 0.54784544, 0.50116331],
       [0.06313018, 0.71396488, 0.60298525]])
```

```python
# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))
```

###### Output


```
array([[ 1.46142422, -0.57102576, -0.85261663],
       [ 0.78455376,  1.52797398,  0.80883395],
       [-0.80088332,  0.06565212, -0.6156856 ]])
```

```python
# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))
```

###### Output


```
array([[6, 3, 3],
       [3, 9, 9],
       [7, 0, 1]])
```

```python
# Create a 3x3 identity matrix
np.eye(3)
```

###### Output


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

###### Output


```
array([1., 1., 1.])
```

## NumPy standard data types

NumPy arrays contain values of a single type, so it is important to have detailed knowledge of those types and their limitations. Because NumPy is built in C, the types will be familiar to users to C.

The standard NumPy data types are listed in the following table. Note that when constructing an array, you can specify them using a string:

```python
np.zeros(10, dtype='int16')
```

###### Output


```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int16)
```

Or using the associated NumPy object:

```python
np.zeros(10, dtype=np.int16)
```

###### Output


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

###### Output


```
x3 ndim: 3
x3 shape: (3, 4, 5)
x3 size: 60
```

Another useful attribute is the `dtype`, the data type of the array:

```python
print('x3 dtype:', x3.dtype)
```

###### Output


```
x3 dtype: int64
```

Other attributes include `itemsize`, which lists the size (in bytes) of each array element, and `nbytes`, which lists the total size (in bytes) of the array:

```python
print('itemsize:', x3.itemsize, 'bytes')
print('nbytes:', x3.nbytes, 'bytes')
```

###### Output


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

###### Output


```
Generalization check result for x1: True
Generalization check result for x2: True
Generalization check result for x3: True
```

### Array Indexing: Accessing Single Elements

In one-dimensional array, you can access the ith value (counting from zero) by specifying the desired index in square brackets, just as with Python lists:

```python
x1
```

###### Output


```
array([5, 0, 3, 3, 7, 9])
```

```python
x1[0]
```

###### Output


```
5
```

```python
x1[4]
```

###### Output


```
7
```

To index from the end of the array, you can use negative indices:

```python
x1[-1]
```

###### Output


```
9
```

```python
x1[-2]
```

###### Output


```
7
```

In multidimensional array, you access items using a comma-separated tuple of indices:

```python
x2
```

###### Output


```
array([[3, 5, 2, 4],
       [7, 6, 8, 8],
       [1, 6, 7, 7]])
```

```python
x2[0, 0]
```

###### Output


```
3
```

```python
x2[2, 0]
```

###### Output


```
1
```

```python
x2[2, -1]
```

###### Output


```
7
```

You can also modify values using any of the above index notation:

```python
x2[0, 0] = 12
x2
```

###### Output


```
array([[12,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])
```

Keep in mind that, unlike Python lists, NumPy arrays have a fixed type. This means, for example, that if you attempt to insert a floating-point value to an integer array, the value will be silently truncated. Don't be caught unaware by this behaviour!

```python
x1[0] = 3.14159  # this will be truncated
x1
```

###### Output


```
array([3, 0, 3, 3, 7, 9])
```

### Array Slicing: Accessing Subarrays

The NumPy slicing syntax follows that of the standard Python list; to access a slice of an array `x`, use this:

    x[start:stop:step]

If any of these are unspecified, they default to the values `start=0`, `stop=size of dimension`, `step=1`.

#### One-dimensional subarrays

```python
x = np.arange(10)
x
```

###### Output


```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

```python
x[:5]  # first five elements
```

###### Output


```
array([0, 1, 2, 3, 4])
```

```python
x[5:]  # elements after index 5
```

###### Output


```
array([5, 6, 7, 8, 9])
```

```python
x[4:7]  # middle subarray
```

###### Output


```
array([4, 5, 6])
```

```python
x[::2]  # every other element
```

###### Output


```
array([0, 2, 4, 6, 8])
```

```python
x[1::2]  # every other element, starting at index 1
```

###### Output


```
array([1, 3, 5, 7, 9])
```

A potentially confusing case is when the `step` value is negative. In this case, the defaults for `start` and `stop` are swapped. This becomes a convenient way to reverse an array:

```python
x[::-1]  # all elements, reversed
```

###### Output


```
array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
```

```python
x[5::-2]  # reverse every other from 5
```

###### Output


```
array([5, 3, 1])
```

#### Multidimensional subarrays

Multidimensional slices work in the same way, with multiple slices separated by commas. For example:

```python
x2
```

###### Output


```
array([[12,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])
```

```python
x2[:2, :3]  # two rows, three columns
```

###### Output


```
array([[12,  5,  2],
       [ 7,  6,  8]])
```

```python
x2[:3, ::2]  # all rows, every other column
```

###### Output


```
array([[12,  2],
       [ 7,  8],
       [ 1,  7]])
```

Finally, subarray dimensions can even be reversed together:

```python
x2[::-1, ::-1]
```

###### Output


```
array([[ 7,  7,  6,  1],
       [ 8,  8,  6,  7],
       [ 4,  2,  5, 12]])
```

#### Accessing array rows and columns

One commonly needed routine is accessing single rows or columns of an array. You can do this by combining indexing and slicing using an empty slice marked by a single colon (:):

```python
x2[:, 0]  # first column of x2
```

###### Output


```
array([12,  7,  1])
```

```python
x2[0, :]  # first row of x2
```

###### Output


```
array([12,  5,  2,  4])
```

In case of row access, the empty slice can be omitted for a more compact syntax:

```python
x2[0]
```

###### Output


```
array([12,  5,  2,  4])
```

#### Subarrays as no-copy views

NumPy return _views_ rather than _copies_ of the array data. This is one area in which NumPy array slicing differs from Python list slicing: in lists, slices will be copies. Consider our two-dimensional array from before:

```python
x2
```

###### Output


```
array([[12,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])
```

Let's extract a 2x2 subarray from this:

```python
x2_sub = x2[:2, :2]
x2_sub
```

###### Output


```
array([[12,  5],
       [ 7,  6]])
```

Now if we modify this subarray, we'll see that the original array is changed! Observe:

```python
x2_sub[0, 0] = 99
x2_sub
```

###### Output


```
array([[99,  5],
       [ 7,  6]])
```

```python
x2
```

###### Output


```
array([[99,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])
```

This default behaviour is actually quite useful: it means that when we work with large datasets, we can access and process pieces of these datasets without the need to copy the underlying data buffer.

#### Creating copies of arrays

Despite the nice feature of array views, it is sometimes useful to instead explicitly copy the data within an array or a subarray. This can be most easily done with the `copy()` method:

```python
x2_sub_copy = x2[:2, :2].copy()
x2_sub_copy
```

###### Output


```
array([[99,  5],
       [ 7,  6]])
```

If we now modify this subarray, the original array is not touched:

```python
x2_sub_copy[0, 0] = 42
x2_sub_copy
```

###### Output


```
array([[42,  5],
       [ 7,  6]])
```

```python
x2
```

###### Output


```
array([[99,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])
```

### Reshaping of Arrays

Another useful type of operation is reshaping of arrays. The most flexible way of doing this is with the `reshape()` method. For example, if you want to put the numbers 1 through 9 in a 3x3 grid, you can do the following:

```python
grid = np.arange(1, 10).reshape((3, 3))
grid
```

###### Output


```
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
```

Note that for this to work, the size of the initial array must match the size of the reshaped array. Where possible, the `reshape` method will use a no-copy view of the initial array, but with non-contiguous memory buffers this is not always the case.

Another common reshaping pattern is the conversion of a one-dimensional array into a two-dimensional row or column matrix. You can do this with the `reshape` method, or more easily by making use of the `newaxis` keyword within a slice operation:

```python
x = np.array([1, 2, 3])

# row vector via reshape
x.reshape((1, 3))
```

###### Output


```
array([[1, 2, 3]])
```

```python
# row vector via newaxis
x[np.newaxis, :]
```

###### Output


```
array([[1, 2, 3]])
```

```python
# column vector via reshape
x.reshape((3, 1))
```

###### Output


```
array([[1],
       [2],
       [3]])
```

```python
# column vector via newaxis
x[:, np.newaxis]
```

###### Output


```
array([[1],
       [2],
       [3]])
```

### Array Concatenation and Splitting

#### Concatenation of arrays

Concatenation, or joining of two arrays in NumPy, is primarily accomplished through the routines `np.concatenate`, `np.vstack`, and `np.hstack`. `np.concatenate` takes a tuple or list of arrays as its first argument, as we can see here:

```python
x = np.array([1, 2, 3,])
y = np.array([3, 2, 1])
np.concatenate([x, y])
```

###### Output


```
array([1, 2, 3, 3, 2, 1])
```

You can also concatenate more than two arrays at once:

```python
z = [99, 99, 99]
np.concatenate([x, y, z])
```

###### Output


```
array([ 1,  2,  3,  3,  2,  1, 99, 99, 99])
```

`np.concatenate` can also be used for two-dimensional arrays:

```python
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
```

```python
# concatenate along the first axis
np.concatenate([grid, grid])
```

###### Output


```
array([[1, 2, 3],
       [4, 5, 6],
       [1, 2, 3],
       [4, 5, 6]])
```

```python
# concatenate along the second axis (zero-indexed)
np.concatenate([grid, grid], axis=1)
```

###### Output


```
array([[1, 2, 3, 1, 2, 3],
       [4, 5, 6, 4, 5, 6]])
```

For working with arrays of mixed dimensions, it can be clearer to use the `np.vstack` (vertical stack) and `np.hstack` (horizontal stack) functions:

```python
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])
```

###### Output


```
array([[1, 2, 3],
       [9, 8, 7],
       [6, 5, 4]])
```

```python
y = np.array([[99],
              [99]])

# horizontally stack the arrays
np.hstack([grid, y])
```

###### Output


```
array([[ 9,  8,  7, 99],
       [ 6,  5,  4, 99]])
```

Similarly, `np.dstack` will stack arrays along the third axis.

#### Splitting of arrays

The opposite of concatenation is splitting, which is implemented by the functions `np.split`, `np.hsplit`, and `np.vsplit`. For each of these, we can pass a list of indices giving the split points:

```python
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
x1, x2, x3
```

###### Output


```
(array([1, 2, 3]), array([99, 99]), array([3, 2, 1]))
```

The second argument of `np.split` is `indices_or_sections`. If `indices_or_sections` is an integer, N, the array will be divided into N equal arrays along `axis`.  If such a split is not possible, an error is raised. If `indices_or_sections` is a 1-D array of sorted integers, the entries indicate where along `axis` the array is split.  For example, ``[2, 3]`` would, for ``axis=0``, result in

  - `ary[:2]`
  - `ary[2:3]`
  - `ary[3:]`

Notice that _N_ split points lead to _N + 1_ subarrays. The related functions `np.hsplit` and `np.vsplit` are similar:

```python
grid = np.arange(16).reshape((4, 4))
grid
```

###### Output


```
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
```

```python
upper, lower = np.vsplit(grid, [2])
upper, lower
```

###### Output


```
(array([[0, 1, 2, 3],
        [4, 5, 6, 7]]),
 array([[ 8,  9, 10, 11],
        [12, 13, 14, 15]]))
```

```python
left, right = np.hsplit(grid, [2])
left, right
```

###### Output


```
(array([[ 0,  1],
        [ 4,  5],
        [ 8,  9],
        [12, 13]]),
 array([[ 2,  3],
        [ 6,  7],
        [10, 11],
        [14, 15]]))
```

Similarly, `np.dsplit` will split arrays along the third axis.

## Computation on NumPy Arrays: Universal Functions

Computation on NumPy arrays can be very fast or it can be very slow. The key to making it fast is to use vectorized operations, generally implemented through NumPy's _universal functions_ (ufuncs).

### Introducing UFuncs

For many types of operations, NumPy provides a convenient interface into just this kind of statically typed, compiled routine. This is known as _vectorized_ operation.

```python
values = np.random.randint(1, 100, size=500000)
print(1.0 / values)
```

###### Output


```
[0.02083333 0.25       0.01298701 ... 0.04761905 0.02439024 0.01639344]
```

Looking at the execution time for our big array, we see that it completes orders of magnitude faster than the ordinary Python loop:

```python
%timeit (1.0 / values)
```

###### Output


```
10.5 ms ± 2.22 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

If we measure the time with an ordinary loop, we will find a huge time difference:

```python
def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
```

```python
%timeit (compute_reciprocals(values))
```

###### Output


```
264 ms ± 50.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Before we see an operation between a scalar and an array, but we can also see _ufuncs_ between multiple arrays:

```python
np.arange(5) / np.arange(1, 6)
```

###### Output


```
array([0.        , 0.5       , 0.66666667, 0.75      , 0.8       ])
```

And _ufunc_ operations are not limited to one dimensional arrays - they can act on multi-dimensional arrays as well:

```python
x = np.arange(9).reshape((3, 3))
2 ** x
```

###### Output


```
array([[  1,   2,   4],
       [  8,  16,  32],
       [ 64, 128, 256]])
```