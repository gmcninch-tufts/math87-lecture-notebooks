---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

Math087 - Mathematical Modeling
===============================
[Tufts University](http://www.tufts.edu) -- [Department of Math](http://math.tufts.edu)  
[George McNinch](http://gmcninch.math.tufts.edu) <george.mcninch@tufts.edu>  
*Fall 2020*

Course material (Week 3): Network flows and linear programming 
---------------------------------------------------------------------------


So far we have looked at a few examples of linear programs. The key step in modeling these
problems is to write down the program itself.

As we saw, for simple programs, such as the carpenter problem, we can figure it out geometrically.
There were only a few variables and a few obvious constraints and it was easy to check all the
“vertices.”

Let's consider a more complex problem.



Restaurant Example
------------------

Suppose that you are opening a new restaurant and need to make sure you have enough clean
tablecloths to meet expected demand in the first week. On each day, you can buy new tablecloths for
\$5. Used tablecloths can be laundered and returned the next day for \$2 or the following day for \$1.

Your expected tablecloth demands are:

| Day                |   1 |   2 |   3 |   4 |   5 |   6 |   7 |
| :--                | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| tablecloths needed |  10 |  10 |  15 |  20 |  40 |  40 |  30 |

Let’s formulate a linear program to minimize the costs.
Variables:

- $b_i$ = # tablecloths bought on day $i$, $1 ≤ i ≤ 7$.
- $f_i$ = # dirty tablecloths sent to fast laundry on day $i$
- $s_i$ = # dirty tablecloths sent to slow laundry on day $i$
- $t_i$ = # tablecloths needed on day $i$.

First, let’s write down the objective (assuming we only care about week 1):

The goal is to minimize the quantity
$$5 \sum_{i=1}^7 b_i + 2\sum_{i=1}^6 f_i + \sum_{i=1}^5 s_i$$

What are the constraints?

- day 1
  - we need enough tablecloths for day 1, so $t_1 \le b_1$
  - we can't clean more than we've used: $f_1 + s_1 \le t_1$.
- day 2
  - demand must be met from purchases on day 2, plus surplus from day 1, plus fast laundry from day 1:
  $$t_2 \le b_2  + (b_1 - t_1) + f_1$$
- day 3
  - demand again met from purchases on day 3, plus leftover from the previous days, plus
those laundered from the fast service on day 2, and those laundered via the slow service on day 1
  $$t_3 \le b_3 + (b_2 + (b_1 − t_1 ) + f_1 ) − t_2 + f_2 + s_1$$

etc.

This becomes increasingly hard to keep track of and formulate.

So, instead, we build what’s called a network model and we track the flow of tablecloths!

Let's draw part of a diagram:

```python
from graphviz import Digraph

## https://www.graphviz.org/
## https://graphviz.readthedocs.io/en/stable/index.html

dot = Digraph('tablecloth network model')

dot.attr(rankdir='LR')
dot.node('s','source of new tablecloths')

with dot.subgraph(name='clean') as c:
    c.attr(rank='same')
    c.node('d1c','day 1 clean')
    c.node('d2c','day 2 clean')
    c.node('d3c','day 3 clean')
    c.node('d4c','day 4 clean')

with dot.subgraph(name='dirty') as c:
    c.attr(rank='same')
    c.node('d1u','day 1 used')
    c.node('d2u','day 2 used')
    c.node('d3u','day 3 used')

dot.edge('s','d1c',label='cost=5')
dot.edge('s','d2c',label='cost=5')
dot.edge('s','d3c',label='cost=5')
dot.edge('s','d4c',label='cost=5')

dot.edge('d1c','d2c',label='cost=0')
dot.edge('d2c','d3c',label='cost=0')
dot.edge('d3c','d4c',label='cost=0')


dot.edge('d1c','d1u',label='cost=0,ℓ=t_1')
dot.edge('d2c','d2u',label='cost=0,ℓ=t_2')
dot.edge('d3c','d3u',label='cost=0,ℓ=t_3')

dot.edge('d1u','d2c',label='cost=2') ## fast laundry
dot.edge('d1u','d3c',label='cost=1') ## slow laundry

dot.edge('d2u','d4c',label='cost=2') ## fast laundry
dot.edge('d3u','d4c',label='cost=1') ## slow laundry

dot
```

(extrapolate the diagram for the remaining days...)

---

How do we make a linear program of this??

The above diagram represents a *directed graph*. The edges in this graph -- i.e. the arrows between nodes -- track the "flow" of tablecloths.

- We introduce a variable for each arrow. The value of the variable represents the number of tablecloths that move from the start to finish of the arrow.

- Some arcs have lower bounds (e.g. $\ell$ = t_1). If no lower bound is mentioned, there is an implied lower bound of 0.

- Some arcs have upper bounds.
  These model maximum supply or throughput. Nothing implies a bound of $\infty$.
  
- Each “internal” node has conservation -- i.e. ``outputs - inputs = 0``.

- Each arc has a cost. The Objective function is the sum of the quantities (arc costs $\times$ flow variable).



Since we are to have one variable for each arrow in the above diagram,

- $b_i$ = # tablecloths bought on day $i, 1 ≤ i ≤ 7$.
- $u_i$ = # tablecloths used on day $i, 1 ≤ i ≤ 7$.
- $c_i$ = # tablecloths carried over from day $i$ to $i + 1$.
- $f_i$ = # dirty tablecloths sent to fast laundry on day $i$
- $s_i$ = # dirty tablecloths sent to slow laundry on day $i$
- $t_i$ = # tablecloths needed on day $i$.

Now the objective equation has the form:

$$5\sum_{i=1}^j b_i + 0 \sum_{i=1}^7 u_i + 0 \sum_{i=1}^7 c_i + 2 \sum_{i=1}^6 f_i + \sum_{i=1}^5 s_i$$

We require $t_i \le u_i$ for $1 \le i \le 7$ (lower bounds). These lower bounds arise from the arrows from "Day i clean" to "Day i used" with the label $\ell = t_i$. 

We impose no upper bounds on the variables.

For each node, we get a conservation equation:

- $u_1 + c_1 − b_1 = 0$ (from "Day 1 clean")
- $s_1 + f_1 − u_1 = 0$ (from "Day 1 used")
- $u_2 + c_2 − b_2 − c_1 − f_1 = 0$ (from "Day 2 clean")

  Notice in the diagram that there are 2 arrows "coming out of" the node "Day 2 clean"
  and 3 arrows "going in". This is reflected in the above equation.

- $s_2 + f_2 − u_2 = 0$ (from "Day 2 used")
- $u_3 + c_3 − b_3 − c_2 − f_2 − s_1 = 0$ (from "Day 3 clean")
- $s_3 + f_3 − u_3 = 0$ (from "Day 3 used")
- and so on...

Remarks:
--------
- There are $6 \times 7 = 42$ variables. So the objective equation will be given be a vector $\mathbf{c} \in \mathbb{R}^{42}$.

- there are 14 "equality constraints" arising from the conservation equation at each node. Thus the equality constraints are given by a $14 \times 42$ matrix.

- there are 7 inequality constraints, given by a $7 \times 42$ matrix.





```python

```
