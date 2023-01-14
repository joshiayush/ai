# Functions

In [mathematics](https://en.wikipedia.org/wiki/Mathematics), a function from a [set](https://en.wikipedia.org/wiki/Set_(mathematics)) $X$ to a set $Y$ assigns to each element of $X$ exactly one element of $Y$. The set $X$ is called the [domain](https://en.wikipedia.org/wiki/Domain_of_a_function) of the function and the set $Y$ is called the [codomain](https://en.wikipedia.org/wiki/Codomain) of the $\mathrm{function.}^\mathrm{[citation\ needed]}$

A function is most often denoted by the letters such as $f$, $g$, and $h$, and the value of the function $f$ at an element $x$ of its domain is denoted by $f(x)$; the numerical value resulting from the _function evaluation_ at a particular input value is denoted by replacing $x$ with this value; for example, the value of $f$ at $x=4$ is denoted by $f(4)$. When the function is not named and is represented by an [expression](https://en.wikipedia.org/wiki/Expression_(mathematics)) $E$, the value of the fuction at, say, $x=4$, may be denoted by $E\vert_{x=4}$. For example, the value at $4$ of the function that maps $x$ to $(x+1)^2$ may be denoted by $(x+1^2)\vert_{x=4}$ $\mathrm{(which\ results\ in\ 25).}^\mathrm{[citation\ needed]}$

A function is uniquely represented by the set of all [pairs](https://en.wikipedia.org/wiki/Pair_(mathematics)) $(x, f(x))$, called the [graph of the function](https://en.wikipedia.org/wiki/Graph_of_a_function), a popular means of illustrating the function. When the domain and the codomain are sets of real numbers, each such pair may be thought of as the [Cartesian coordinates](https://en.wikipedia.org/wiki/Cartesian_coordinates) of a point in the plane.

## Definition

A function from a [set](https://en.wikipedia.org/wiki/Set_(mathematics)) of $X$ to a set of $Y$ is an assignment of an element of $Y$ to each element of $X$. The set $X$ is called the [domain](https://en.wikipedia.org/wiki/Domain_of_a_function) of the function and the set $Y$ is called the [codomain](https://en.wikipedia.org/wiki/Codomain) of the function.

A function, its domain, and its codomain, are declared by the notation $f:X \to Y$, and the value of a function $f$ at an element $x$ of $X$, denoted by $f(x)$, is called the image of $x$ under $f$, and the value of $f$ applied to the argument $x$.

Functions are also called [maps](https://en.wikipedia.org/wiki/Map_(mathematics)) or mappings.

Two functions $f$ and $g$ are equal if their domain and codomain sets are the same and their output value agree on the whole domain. More formally, given $f:X \to Y$ and $g:X \to Y$, we have $f = g$ if and only if $f(x)=g(x)$ for all $x \in X$.

## Notation

There are various standard ways for denoting functions. The most commonly used notation is functional notation, which is the first notation described below.

### __Functional Notation__

In functional notation, the function is immediately given a name, such as $f$, and its definition is given by what $f$ does to the explicit argument $x$, using a formula in terms of $x$. For example, the function which takes a real number as input and outputs that number plus $1$ is denoted by

$f(x) = x+1$

If a function is defined in this notation, its domain and codomain are implicitly taken to both be $\mathbb{R}$, the set of real numbers. If the formula cannot be evaluated at all real numbers, then the domain is implicitly taken to be the maximal subset of $\mathbb{R}$ on which the formula can be evaluated; see [Domain of a function](https://en.wikipedia.org/wiki/Domain_of_a_function).

A more complicated example is the function

$f(x)= \mathrm{sin}(x+1)^2$

In this example, the function $f$ takes a real number as input, squares it, then adds $1$ to the result, then takes the $sine$ of the result, and returns the final result as the output.

### __Arrow Notation__

Arrow notation defines the rule of a function inline, without requiring a name to be given to the function. For example, $x \mapsto x+1$ is the function which takes a real number as input and outputs that number plus $1$. Again a domain and codomain of $\mathbb{R}$ is implied.

The domain and codomain can also be explicitly stated, for example:

```math
\begin{aligned}
\operatorname{sqr} \colon \mathbb{Z} & \to \mathbb{Z} \\
x & \mapsto x^{2}.
\end{aligned}
```

This defines a function $sqr$ from the integers to the integers that returns the square of its input.

As a common application of the arrow notation, suppose $f \colon X \times X \to Y; \;(x,t) \mapsto f(x,t)$ is a function in two variables, and we want to refer to a [partially applied function](https://en.wikipedia.org/wiki/Partial_application) $X \to Y$ produced by fixing the second argument to the value $t_{0}$ without introducing a new function name. The map in question could be denoted $x \mapsto f(x,t_{0})$ using the arrow notation. The expression $x \mapsto f(x,t_{0})$ (read: "the map taking $x$ to $f(x, t_{0})$") represents this new function with just one argument, whereas the expression $f(x_{0}, t_{0})$ refers to the value of the function $f$ at the point $(x_{0}, t_{0})$.

### __Index Notation__

Index notation is often used instead of functional notation. That is, instead of writing $f(x)$, one writes $f_{x}$.

This is typically the case for functions whose domain is the set of the [natural numbers](https://en.wikipedia.org/wiki/Natural_number). Such a function is called a [sequence](https://en.wikipedia.org/wiki/Sequence_(mathematics)), and, in this case the element $f_{n}$ is called the $nth$ element of the sequence.

The index notation is also often used for distinguishing some variables called [parameters](https://en.wikipedia.org/wiki/Parameter) from the "true variables". In fact, parameters are specific variables that are considered as being fixed during the study of a problem. For example, the map $x \mapsto f(x,t)$ would be denoted $f_{t}$ using index notation, if we define the collection of maps $f_{t}$ by the formula $f_{t}(x)=f(x, t)$ for all $x, t \in X$.

### __Dot Notation__

In the notation $x \mapsto f(x)$, the symbol $x$ does not represent any value, it is simply a [placeholder](https://en.wikipedia.org/wiki/Placeholder_name) meaning that, if $x$ is replaced by any value on the left of the arrow, it should be replaced by the same value on the right of the arrow. Therefore, $x$ may be replaced by any symbol, often an [interpunct](https://en.wikipedia.org/wiki/Interpunct) "$⋅$". This may be useful for distinguishing the function $f(⋅)$ from its value $f(x)$ at $x$.

For example, $a(.)^2$ may stand for the function $x \mapsto ax^2$, and $\int_a^\mathrm{(.)} f(u) \: \mathrm{d}u$ may stand for a function defined by an integral with variable upper bound: $x \mapsto \int_a^x f(u) \: \mathrm{d}u$.

## Domain of a function

In [mathematics](https://en.wikipedia.org/wiki/Mathematics), the __domain__ of a [function](https://en.wikipedia.org/wiki/Function_(mathematics)) is the [set](https://en.wikipedia.org/wiki/Set_(mathematics)) of inputs accepted by the function. It is sometimes denoted by $\operatorname{dom}(f)$ or $\operatorname{dom} f$, where $f$ is the function.

More precisely, given a function $f: X \to Y$, the domain of $f$ is $X$. Note that in modern mathematical language, the domain if part of the definition of a function rather than a property of it.

In the special case that $X$ and $Y$ are both subsets of $\mathbb{R}$, the function $f$ can be graphed in the [Cartesian coordinate system](https://en.wikipedia.org/wiki/Cartesian_coordinate_system). In this case, the domain is represented on the $x- \mathrm{axis}$ of the graph, as the projection of the graph on the function onto the $x- \mathrm{axis}$.

For a function $f: X \to Y$, the set $Y$ is called the [codomain](https://en.wikipedia.org/wiki/Codomain), and the set of values attained by the function (which is a subset of $Y$) is called its [range](https://en.wikipedia.org/wiki/Range_of_a_function) or [image](https://en.wikipedia.org/wiki/Image_(mathematics)).

Any function can be restricted to a subset of its domain. The [restriction](https://en.wikipedia.org/wiki/Restriction_(mathematics)) of $f: X \to Y$ to $A$, where $A \subseteq X$, is written as $f \vert_{A}: A \to Y$.

### Natural Domain

If a [real function](https://en.wikipedia.org/wiki/Real_function) $f$ is given by a formula, it may be not defined for some values of the variable. In this case, it is a [partial function](https://en.wikipedia.org/wiki/Partial_function), and the set of real numbers on which the formula can be evaluated to a real number is called the __natural domain__ or __domain of definition__ of $f$. In many contexts, a partial function is called simply a _function_, and its natural domain is called simply its _domain_.

### __Examples__

* The function $f$ defined by $f(x) = \dfrac{1}{x}$ cannot be evaluated at $0$. Therefore, the natural domain of $f$ is the set of natural numbers excluding $0$, which can be denoted by $\mathbb{R} \setminus \{0\}$ or $\{x \in \mathbb{R} : x \neq 0\}$.
* The [piecewise](https://en.wikipedia.org/wiki/Piecewise) function $f$ defined by $f(x)=$
```math
\begin{cases}
\dfrac{1}{x} & \quad x \neq 0 , \\
0 & \quad x = 0
\end{cases}
```
  , has as its natural domain the set $\mathbb{R}$ of real numbers.
* The [square root](https://en.wikipedia.org/wiki/Square_root) function $f(x) = \sqrt{x}$, has as its natural domain the set of non-negative real numbers, which can be denoted by $\mathbb{R}_{\geq 0}$, the interval $[0, \infty)$, or $\{x \in \mathbb{R} : x \geq 0\}$.
* The [tangent function](https://en.wikipedia.org/wiki/Tangent_function), denoted $\mathrm{tan}$, has as its natural domain the set of all real numbers which are not of the form $\dfrac{\pi}{2} + k \pi$ for some [integer](https://en.wikipedia.org/wiki/Integer) $k$, which can be written as $\mathbb{R} \setminus \{\dfrac{\pi}{2} + k \pi : k \in \mathbb{Z}\}$.