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

# Multi-variable optimization

<!-- #region -->

## Optimization with functions of several variables

Consider a function \\(f(x,y)\\) of two variables. You learned in
Calculus 3 (*vector calculus*) how to search for the points \\((x,y)\\) at which
\\(f\\) assumes its maximum and minimum value. Let's briefly recall
this story.

Recall that for a function of a single variable, critical points are those points for which the tangent line is horizontal. In the single variable case, the criteria depends instead on the *tangent plane*.

Recall that the [**normal vector**](https://en.wikipedia.org/wiki/Normal_(geometry)) at the point \\(P=(x_0,y_0,f(x_0,y_0))\\) to the surface defined by \\(z = f(x,y)\\) is given by
\\[\mathbf{n}\vert_P = \left ( \mathbf{i} + \dfrac{\partial f}{\partial x} \mathbf{k} \right )_P \times 
                        \left ( \mathbf{j} + \dfrac{\partial f}{\partial y} \mathbf{k} \right )_P 
                     = \left ( \mathbf{k}  - \dfrac{\partial f}{\partial x} \mathbf{i}                                                                                - \dfrac{\partial f}{\partial y} \mathbf{j} \right )_P\\]
  
Now, the **tangent plane** at \\(P\\) to the surface \\(z = f(x,y)\\) is just
the plane orthogonal to this normal vector \\(\mathbf{n}_P\\). Thus, the tangent plane at 
\\(P\\) is horizontal just in case this normal vector points in the \\(\mathbf{k}\\) 
direction -- i.e. provided that
\\[(\clubsuit) \quad \dfrac{\partial{f}}{\partial{x}} \bigg\vert_{(x_0,y_0)} = 0 \quad
\text{and} \quad  \dfrac{\partial{f}}{\partial{y}} \bigg\vert_{(x_0,y_0)} = 0\\]
  
Just as in the one variable case, the points \\((x_0,y_0)\\) for which the tangent plane to  the surface \\(P=(x_0,y_0,f(x_0,y_0))\\) is horizontal are the *critical points*.
  
So we find the critical points by simultaneously solving the euqations\\((\clubsuit)\\).

There is a *second derivative test* which gives information about the "max/min status" of these critical points.
  
To use this test, consider the matrix of second partial derivatives
  \\[ M(x_0,y_0) = \begin{pmatrix} 
        \dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x \partial y} \\
        \dfrac{\partial^2 f}{\partial y \partial x} & \dfrac{\partial^2 f}{\partial y^2} \\
      \end{pmatrix} \Bigg\vert_{(x_0,y_0)}.\\]
      
For reasonable functions, the "mixed partials" \\(\dfrac{\partial^2 f}{\partial x \partial y}\\) and \\(\dfrac{\partial^2 f}{\partial y \partial x}\\) coincide.
  
The determinant of a \\(2 \times 2\\) matrix \\(\begin{pmatrix} a & b \\ c & d \end{pmatrix}\\) is \\(ad - bc\\)).
  
So, the *determinant* of \\(M\\) is the expression
  
\\[D=D(x_0,y_0) = \left(\dfrac{\partial^2 f}{\partial x^2}\cdot \dfrac{\partial^2 f}{\partial y^2}
 - \left[\dfrac{\partial^2 f}{\partial x \partial y}\right]^2\right) \bigg\vert_{(x_0,y_0)} \\]
   
   
Suppose that \\((x_0,y_0)\\) is a critical point. 
- If \\(D>0\\) and \\(\dfrac{\partial f}{\partial x}\bigg\vert_{(x_0,y_0)}<0\\), then \\(f(x,y)\\) has a relative maximum at \\((x_0,y_0)\\).
- If \\(D>0\\) and \\(\dfrac{\partial f}{\partial x}\bigg\vert_{(x_0,y_0)}>0\\), then \\(f(x,y)\\) has a relative minimum at \\((x_0,y_0)\\).

- If \\(D<0\\), then \\(f(x,y)\\) has a saddle point at \\((x_0,y_0)\\).
- If \\(D=0\\), the second derivative test is inconclusive. 



<!-- #endregion -->

<!-- #region -->
## Lagrange multipliers

Consider a function \\(f(x,y)\\) of two variables. We are interested here in finding maximal or minimal values of \\(f\\) subject to a *constraint*. The sort of constraint we have in mind is a restriction on the possible pairs \\((x,y)\\) -- so we have a second function \\(g(x,y)\\) and we want to maximize (or minimize) \\(f\\) subject
to the condition that \\(g(x,y) = c\\) for some fixed quantity \\(c\\).

We introduce a "new" function -- now of *three* variables - known as the **Lagrangian**. It is given by the formula
\\[F(x,y,\lambda) = f(x,y) - \lambda \cdot (c-g(x,y))\\]

We can calculate the *partial derivatives* of this Lagrangian; they are:

\\[\dfrac{\partial F}{\partial x} = \dfrac{\partial f}{\partial x} - \lambda\dfrac{\partial g}{\partial x}\\]

\\[\dfrac{\partial F}{\partial y} = \dfrac{\partial f}{\partial y} - \lambda\dfrac{\partial g}{\partial y}\\]

\\[\dfrac{\partial F}{\partial \lambda} = c-g(x,y)\\]

If we seek critical points of the Lagrangian, we find that 
\\[0 = \dfrac{\partial F}{\partial x} = \dfrac{\partial f}{\partial x} - \lambda\dfrac{\partial g}{\partial x}\\]
and similarly for \\(y\\), so that
\\[ \dfrac{\partial f}{\partial x} = \lambda \dfrac{\partial g}{\partial x} \quad \text{and}\quad
\dfrac{\partial f}{\partial x} = \lambda \dfrac{\partial g}{\partial x}\\]
i.e.
\\[ \left (\dfrac{\partial f}{\partial x} \mathbf{i} + \dfrac{\partial f}{\partial y} \mathbf{j} \right)
= \lambda \left (\dfrac{\partial g}{\partial x} \mathbf{i} + \dfrac{\partial g}{\partial y} \mathbf{j} \right)\\]

(Recall that \\(\dfrac{\partial f}{\partial x} \mathbf{i} + \dfrac{\partial f}{\partial y} \mathbf{j}\\) is the
*gradient* \\(\nabla f\\) of \\(f\\)).

Moreover, we find that
\\[0 = \dfrac{\partial F}{\partial \lambda} = c - g(x,y).\\]

Summarizing, the condition that \\((x_0,y_0,\lambda_0)\\) is a critical point of \\(F\\) is equivalent to two requirements: 
- \\((x_0,y_0)\\) must be on the level curve \\(g(x,y) = c\\), and
- the gradient vectors must satisfy \\(\nabla f \vert_{(x_0,y_0)} = \lambda_0 \nabla g \vert_{(x_0,y_0)}\\).

The crucial point is: optimal values for \\(f\\) along the level curve \\(g(x,y) = c\\) will be found among the critical points of \\(F\\). 

Indeed. suppose \\((x_0,y_0)\\)
is a point on the level curve at which \\(f\\) takes its max (or min) value (on the level surface).
We need to argue that the gradient vector \\(\nabla f \vert_{(x_0,y_0)}\\) is "parallel" to the gradient
vector \\(\nabla g \vert_{(x_0,y_0)}\\).

More precisely, we can write \\(\nabla g \vert_{(x_0,y_0)} = \mathbf{v} + \mu \nabla f \vert_{(x_0,y_0)}\\)
for a vector \\(\mathbf{v}\\) perpendicular to \\(\nabla f \vert_{(x_0,y_0)}\\) (and for some scalar \\(\mu\\)).
And we must argue that \\(\mathbf{v}\\) is zero.

But if \\(\mathbf{v}\\) is non-zero, then walking along the level curve \\(g(x,y) = c\\) "in the direction of \\(\mathbf{v}\\)" the  

```python

```
<!-- #endregion -->

```python

```
