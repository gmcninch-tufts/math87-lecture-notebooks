---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---
{{{abstract}}}
<!-- #region -->

# <center> **Week 2**: Root finding algorithms </center>

# <center> Root finding algorithms </center>

We are going to describe some algorithms for finding solutions to
(non-linear) equations $$f(x) = 0$$ The various methods we describe
have different assumptions and requirements.

By a root of $f$, we just mean a real number $x_0$ such that $f(x_0) =
0$.

Of course, for some very special functions $f$, we have formulas for
roots. For example, if \\(f\\) is a *quadratic polynomial*, say $f(x)
= ax^2 + bx + c$ for real numbers $a,b,c$, then there are in general
two roots, given by the formula $$x_0 = \dfrac{-b \pm \sqrt{b^2 -
4ac}}{2a}$$ (of course, these roots are only *real number* in case of
$b^2 - 4ac \ge 0$). But such a formula is far too much to ask for, in
general!

There are algorithmic methods for *approximating* roots of "nice
enough" functions. In the next section, we are going to describe - and
implement with some basic code - one such method (*"bisection"*). Then
we’ll describe a more general function from python’s `scipi` API which
implements various root finding



## Bisection

The [bisection algorithm](https://en.wikipedia.org/wiki/Bisection_method) permits one
to approximate a root of a continuous function \\(f\\), provided that
one knows points \\(x_L < x_R\\) in the domain of \\(f\\) for which
the function values $f(x_L)$ and $f(x_R)$ are non-zero and have
opposite signs. The algorithm then returns an approximate root in the
interval \\((x_L,x_R)\\).

Of course, for a continuous \\(f\\) the [*intermediate value
theorem*](https://en.wikipedia.org/wiki/Intermediate_value_theorem)
implies that there is at least one root $x_0$ of $f$ in the interval
$(x_L,x_R)$.

To find a root, the algorithm iteratively divides the interval
$[x_L,x_R]$ into two sub-intervals by introducing the midpoint $x_C =
\dfrac{x_L + x_R}{2}$. It examines the signs of the values $f(x_L)$,
$f(x_C)$ and $f(x_R)$ and discards the interval on which the sign
doesn't change. (Of course, if $f(x_C)$ happens to be zero, that is
the root!)

So for example, if $f(x_L)$ and $f(x_C)$ differ in sign, the procedure
is repeated on this smaller interval $[x_L,x_C]$.

One specifies in advance the number $N$ of iterations of this
algorithm. After $N$ iterations, the algorithm returns the midpoint of
the interval, and that number is taken as the approximate root.

Writing $x_N$ for the approximate solution returned by the algorithm
after $N$ iterations, one knows that the limit $$\lim_{N \to \infty}
x_N$$ exists -- in words: the estimates converge to a solution.


```python


def bisection(f,a,b,tol=1e-10,max_iterations=10000):
    #    Approximate solution of f(x)=0 on interval [a,b] by bisection method.
    if f(a)*f(b) >= 0:
        print("Bisection method can't be applied because f(a) & f(b) don't differ in sign.")
        return None
    x_L = a
    x_R = b
    n=0
    finished = False
    while not finished:
        x_C = (x_L + x_R)/2       # compute the midpoint of the relevant interval
        if f(x_C) == 0:
            print("Found exact solution.")
        elif f(x_L)*f(x_C) < 0:
            x_L = x_L
            x_R = x_C
        elif f(x_R)*f(x_C) < 0:
            x_L = x_C
            x_R = x_R
        else:
            print("Bisection method fails -- is f continuous?")
            return None
        n = n + 1
        finished = (n >= max_iterations) or abs(f(x_C)) < tol
    return x_C
```

We can use ‘bisection‘ to approximate the roots of \\(f(x) = x^2 - x -1\\). 
Recall that those roots are
\\[\dfrac{1 \pm \sqrt{5}}{2}\\]

```python
def f(x): return x**2 - x -1

[ bisection(f,1,2) , bisection(f,-2,0)]

```

```python
import numpy as np

## here are python's built-in approximations of these irrationalities
[(1 + np.sqrt(5))/2, (1 - np.sqrt(5))/2]
```

We can estimate zeros of the \\(\sin\\) function - here we get an approximation to \\(\pi\\):

```python
import numpy as np

def f(x): return np.sin(x)
[bisection(f,1,4), np.pi]
```

And we can estimate the transcendental number \\(e = \exp(1)\\) e.g. by finding roots of the function \\(f(x) = 1 - \ln(x)\\):

```python
import numpy as np

def f(x): return 1 - np.log(x)

[bisection(f,1,3),np.exp(1)]
```

```python

```

<!-- #region -->
## Secant Method


[Here is the wikipedia description](https://en.wikipedia.org/wiki/Secant_method) of the secant method.

The secant method is very similar to the bisection method except instead of dividing each interval by choosing the midpoint the secant method divides each interval by the secant line connecting the endpoints. The secant method always converges to a root of provided that is continuous on and .

<!-- #endregion -->

```python

## Newton's Method

def newton(f,df,x0,tol=1e-10,max_iterations=1000):

    n = 0
    x = x0

    while n<max_iterations and abs(f(x)) > tol:
        n = n+1;
        if df(x) == 0:
            print("failure: horizontal tangent line")
            return None
        else:
            x = x - f(x)/df(x);
    return {"solution":x,"iterations":n}

# Examples
# --------
#
def f(x):
    return x**2 - x -1

def df(x):
    return 2*x - 1

#  Alternatively, we could have specified  
#  f = lambda x: x**2 - x - 1

newton(f,df,1)

def f(x):
    return (2*x - 1)*(x-3)

def df(x):
    return 2*(x-3) + (2*x-1)
# alt: f = lambda x: (2*x - 1)*(x - 3)

newton(f,df,.9)

import numpy as np

def f(x):
    return np.sin(x)

def df(x):
    return np.cos(x)

newton(f,df,np.pi/2)

def f(x):
    return x**3 - x**2 - 1

def df(x):
    return 3*x**2 - 2*x

newton(f,df,1)

def newton_alt(f,x0,tol=1e-10,max_iterations=1000):

    def df(x):
        epsilon = 1e-7
        return (f(x+epsilon) - f(x))/epsilon
    
    n = 0
    x = x0

    while n<max_iterations and abs(f(x)) > tol:
        n = n+1;
        if df(x) == 0:
            print("failure: horizontal tangent line")
            return None
        else:
            x = x - f(x)/df(x);
    return {"solution":x,"iterations":n}

def f(x):
    return np.sin(x)

newton_alt(f,np.pi/2)

def f(x):
    return x**3 - x**2 - 1

newton_alt(f,.9)



```

## Newton's Method


## Halley's Method
