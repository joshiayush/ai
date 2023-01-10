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

$
f(x) = x+1
$

If a function is defined in this notation, its domain and codomain are implicitly taken to both be $\mathbb{R}$, the set of real numbers. If the formula cannot be evaluated at all real numbers, then the domain is implicitly taken to be the maximal subset of $\mathbb{R}$ on which the formula can be evaluated; see [Domain of a function](https://en.wikipedia.org/wiki/Domain_of_a_function).

A more complicated example is the function

$
f(x)= \mathrm{sin}(x+1)^2
$

In this example, the function $f$ takes a real number as input, squares it, then adds $1$ to the result, then takes the $sine$ of the result, and returns the final result as the output.

### __Arrow Notation__

Arrow notation defines the rule of a function inline, without requiring a name to be given to the function. For example, $x \mapsto x+1$ is the function which takes a real number as input and outputs that number plus $1$. Again a domain and codomain of $\mathbb{R}$ is implied.

The domain and codomain can also be explicitly stated, for example:

$
\begin{aligned}
\operatorname{sqr} \colon \mathbb{Z} & \to \mathbb{Z} \\
x & \mapsto x^{2}.
\end{aligned}
$

This defines a function $sqr$ from the integers to the integers that returns the square of its input.

As a common application of the arrow notation, suppose $f \colon X \times X \to Y; \;(x,t) \mapsto f(x,t)$ is a function in two variables, and we want to refer to a [partially applied function](https://en.wikipedia.org/wiki/Partial_application) $X \to Y$ produced by fixing the second argument to the value $t_{0}$ without introducing a new function name. The map in question could be denoted $x \mapsto f(x,t_{0})$ using the arrow notation. The expression $x \mapsto f(x,t_{0})$ (read: "the map taking $x$ to $f(x, t_{0})$") represents this new function with just one argument, whereas the expression $f(x_{0}, t_{0})$ refers to the value of the function $f$ at the point $(x_{0}, t_{0})$.