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

# Optimization -- Week 1

<!-- #region -->
# Optimization (overview)

---
Optimization is the most common application of mathematics. Here are some “real-world” examples:

- **Business optimization**. A business manager attempts to understand and control parameters in order to maximize profit and minimize costs.

- **Natural resource management**. Control harvest rates to maximize long-term yield, while conserving resources.

- **Environmental regulation**. Governments sets standards to minimize environmental costs, while maximizing production of goods.

- **IT management**. Computer system managers try to maximize throughput and minimize delays.

- **Pharmaceutical optimization**. Doctors and pharmacists regulate drugs to minimize harmful side effects and maximize healing.

---
In this first part of our modeling course, we are going to discuss some sorts of optimization problems and related matters:

- *single variable optimization* and *sensitivity analysis*

- *multivariable optimization*

- *multivariable optimization with constraints*


---



# Single Variable Optimization

In this first section of our modeling class, we are going to examine a few *single variable* optimization problems. In some sense, these amount of complicated versions of *word problems* that you might have met in Calculus I (differential calculus).

The calculus based solution can then be described roughly as follows: 

- find the function \\(f(x)\\) that measures the quantity that you desire to optimize, and the relevant interval \\([a,b]\\) of values of independent variable \\(x\\). 
- find the critical points \\(c_1,c_2,\dots,c_N\\) of \\(f\\) in the interval \\((a,b)\\).
- if \\(f\\) is a *nice enough* function, the maximum and minimum value of \\(f\\) will be found in the list
\\(f(a),f(c_1),\dots,f(c_N),f(b)\\) -- remember that you must check the endpoints \\(a,b\\)!
<!-- #endregion -->

<!-- #region -->



## Example: Oil spill 

> An oil spill has contaminated 200 miles of Alaskan shoreline. The
> shipping company responsible for the accident has been given 14 days
> to clean up the shoreline, after which it will be required to pay a
> fine of \\$10,000 per day for each day in which any part of the
> shoreline remains uncleaned.

*(Let's assume that the fine depends on fractional days, so that
e.g. if the work is completed in 15.5 days, the company would pay a
fine of \\(\dfrac{3}{2}\cdot 10000 = 15000.\\) I'll comment on this
again below)*

> Cleanup crews can be hired, and each crew cleans 5 miles of beach per week.
> 
> There is one local cleanup crew available at a cost of \\$500 per day. 
>
> Additional non-local crews can be hired. The hire of each non-local
> crew incurs an \\$18,000 one-time travel cost. These additional
> crews work for \\$800 per day for each crew (and each crew has the
> same cleanup rate of 5 miles of beach per week).

---

Relevant parameters:

* miles cleaned per crew per week = \\(5\\), so
* \\(m\\) = miles cleaned per crew per day = \\(5/7\\)
* \\(f\\) = fine charged per day = \\$\\(10,000\\)
* \\(TC\\) = travel costs per outside crew = \\$\\(18,000\\)


The main choice that the company must make is: "how many outside crews to hire?"

* \\(n\\) = # of outside crews to hire

According to the background description, there are a number of quantities that depend on this choice:

* \\(t\\) = # of days for complete cleanup
* \\(F\\) = fine to be paid
* \\(C_{crew}\\) = payments to cleanup crews
* \\(C_{tot}\\) = total cleanup cost = \\(F + C_{crew}\\)
   
---

Let's give **mathematical expressions** for these quantities:

* \\(t = 200\cdot \dfrac{1}{n+1} \cdot \dfrac{1}{m}\\)

(Indeed, \\(n+1\\) crews working at a rate of \\(m\\) miles per day
will clean 200 miles of beach in the indicated number of days)

* \\(F= \begin{cases} 0 & \text{if $t<14$} \\ f\cdot(t-14) & \text{if
  $t \ge 14$} \end{cases} \\)

(Indeed, no fine if work is completed within two weeks; otherwise, the
fine is given by the indicated formula)

* \\(C_{crew} = 500\cdot t + 800\cdot t\cdot n + TC\cdot n\\)

* \\(C_{tot} = F + C_{crew}\\)
    
---	
	
Now let's give `python` code for computing these quantities.

Note that we use the parameter values as *default values* for some of the arguments
of the *crew_cost* and *cost* functions.
<!-- #endregion -->

```python
def time(n,miles,cleanup_rate): 
    # time in days required for cleanup 
    # by *n* *non-local* cleanup crews 
    #  (together with the one local crew)
    # of *miles* of shoreline, 
    # where each crew works at the indicated cleanup-rate
    return miles/((n+1)*cleanup_rate)

def F(t,fine_per_day): 
    # The total fine imposed. Depends on:
    # t = # of days for complete cleanup, and 
    # fine = daily fine imposed. 
    return 0 if (t<14) else fine_per_day*(t-14)

def crew_cost(n,miles=200,cleanup_rate=5.0/7,tc=18000):
    # cost in payments to crews. Depends on
    # n = number of non-local crews hired
    # cleanup_rate, and 
    # tc_rate = travel costs per non-local crew
    t=time(n,miles,cleanup_rate)
    return 500*t + 800*t*n + tc*n

def cost(n,fine_per_day=10000,miles=200,cleanup_rate=5.0/7,tc=18000):
    t=time(n,miles,cleanup_rate)
    return F(t,fine_per_day) + crew_cost(n,miles,cleanup_rate,tc) 
```

### Let's first just make a table of results

In our table, the rows will contain the values of the various
quantities for possible values of \\(n\\), the number of "outside"
cleanup crews hired.

For this, we are going to use python's [**Pandas**
module](https://pandas.pydata.org/docs/index.html).  We'll use the
"DataFrame" data structure (which is a bit like a python dictionary
for which the keys are the column headers and the values are the
column data).

```python
import pandas as pd

## The following overrides the usual display formatting of floating point numbers. 
## It is just an aesthetic choice...

pd.set_option('display.float_format', lambda x: "{:,.2f}".format(x))
```

```python
def oil_spill_costs(crew_range   = range(0,25),
                    miles        = 200,
                    fine_per_day = 10000,
                    cleanup_rate = 5.0/7,
                    tc=18000):
    return pd.DataFrame(
            {'#crews' : crew_range,
             'cost'   : map( lambda n: cost(n,fine_per_day,miles,cleanup_rate,tc) , crew_range),
             'days'   : map( lambda n: time(n,miles,cleanup_rate)                 , crew_range),
             'fine'   : map( lambda n: F(time(n,miles,cleanup_rate),fine_per_day) , crew_range)
            },
            index=crew_range)
    
```

```python
oil_spill_costs()  ## Compute use the *default* parameter values.
```

We can of course just scan the columns with our eyes to see where the costs are minimized. But we can also use *pandas* API-functions.

In the terminology of *pandas*, we'll extract the *costs* column `df['cost']` of the "dataframe" `df` as a *series*, and then use the `idxmin` method to find the *index* `j` at which the costs are minimized.
Finally, the loc property of `df` allows to select the data `df.loc[j]` in the row with index label `j`.

```python
df = oil_spill_costs()

df.loc[ df['cost'].idxmin() ]
```

### <a id='minimal_cost_cell'> Minimal costs (for basic parameters)</a>

From the preceding calculation, it appears that the cost is minimized by hiring \\(n=11\\) outside crews. With that number of crews, cleanup takes slightly more than 3 weeks with a total cost of \\$508K (including a fine of \\$93K).

Below, we'll use some calculus to confirm this observation!!!

But first, notice that it is easy to change parameters. For example, let's see what would happen if the fine were doubled and if instead 250 miles of coast were contaminated.

```python
df=oil_spill_costs(miles=250,fine_per_day=20000)

df.loc[ df['cost'].idxmin() ]
```

And here is what happens if the crews are actually able to clean up the beach at a rate of 1 mile/day, instead:

```python
df=oil_spill_costs(cleanup_rate=1.0)

df.loc[ df['cost'].idxmin() ]
```

<!-- #region -->
## <a id="calculus_cell">Applying calculus to the problem</a>

We are going to do our analysis with the "default" values \\(m = 5.0/7\\), \\(TC=18000\\), \\(f=10000\\), for 200 miles of coast.

Recall the formulas:

* \\(t = 200\cdot \dfrac{1}{n+1} \cdot \dfrac{1}{5/7} = \dfrac{280}{n+1}\\)

* \\(F= \begin{cases}  0 &  \text{if $t<14$} \\ 10000\cdot(t-14) & \text{if $t \ge 14$} \end{cases} \\)

* \\(C_{crew} = 500\cdot t + 800\cdot t\cdot n + 18000\cdot n\\)

* \\(C_{tot} = F + C_{crew}\\)

\\(C_{tot}\\) is expressed here as a function of both \\(t\\) and \\(n\\). But of course, \\(t\\) is determined by \\(n\\).

We want to express \\(C_{tot}\\) as a function only of \\(n\\). Of course, the obstacle here is that the fine \\(F\\) is not expressed directly as a function of \\(n\\), and the best way to deal with this is to consider different cases.

We first ask the question: "how many crews would we need if we were to clean
everything up in exactly 14 days?"

For this we must solve the equation \\(t(n)=14\\); i.e.: 
\\[14 = \dfrac{280}{1+n}\\]

Thus, \\(n+1=\dfrac{280}{14}\\). We find that
\\(n+1 = 20\\) so that \\(n=19\\). In other words, if 19 external crews are hired, work is completed in two weeks.

Thus we see that for \\(n \ge 19\\) we have \\(F = 0\\) and \\(C_{tot} = C_{crew}\\), while for \\(n < 19\\)
\\[F(n) = 10000\cdot \left (\dfrac{280}{1+n} - 14\right)\\] 

The remaining expenses are the costs associated with hiring cleanup crews. They are
given by the function:

\\[C_{crew}(n) = \dfrac{500 \cdot 280}{1+n}+\dfrac{800 \cdot 280}{1+n}⋅n+18000⋅n \\]

And, the total cost function is given as a function of \\(n\\) by:
\\[C_{tot}(n) =  \left \{ \begin{matrix} F(n) + C_{crew}(n) & n < 19 \\ C_{crew}(n) & n \ge 19\end{matrix} \right .\\] 


Of course, in python we find these functions as follows:

* `fine(n)   = f(time(n,miles=200,cleanup_rate=5.0/7),fine_per_day=10000)`
* `c_crew(n) = crew_cost(n,miles=200,cleanup_rate=5.0/7,tc=18000)`
* `c_tot(n)  = c_crew(n) + fine(n)`
<!-- #endregion -->

```python
def fine(n):   return F(time(n,miles=200,cleanup_rate=5.0/7),fine_per_day=10000)
def c_crew(n): return crew_cost(n,miles=200,cleanup_rate=5.0/7,tc=18000)
def c_tot(n):  return c_crew(n) + fine(n
                                      )
```

### Graphs

We can use python's [matplotlib](https://matplotlib.org/) package to draw graphs of the functions
`c_tot` and `fine`.


```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(2, 30, 200)

fig, ax = plt.subplots(figsize=(12,6))  
ax.plot(x,np.array([c_tot(n) for n in x]),label="C_tot(n)")
ax.plot(x,np.array([fine(n)  for n in x]),label="Fine(n)")

ax.set_xlabel("# crews")
ax.set_ylabel("total cost in $")
ax.legend()

ax.axvline(x=19,                   ## where the "fine-graph" bifurcates
           color="red",   
           dashes=[1,4])            
ax.axvline(x=11.25,                ## we'll see (below) that the min of C occurs here.
           color="green", 
           dashes=[1,4])            
ax.set_title("Cleanup costs")
pass
```

### Some symbolic calculations

It is not really difficult to differentiate the functions \\(F(n)\\) (for \\(0 \le n \le 19\\)),  and \\(C_{crew}(n)\\) [described above](#calculus_cell) by hand. But let's see how we could do it using symbolic calculations in python's [symbolic mathematics package](https://www.sympy.org/en/index.html).

We will make a "symbolic variable" we'll call `y`.

Now we can make a symbolic version `s_cost_low` of \\(C_{tot}(n)\\) by evaluating \\(n\\) at `y`.

We'll write:

`s_cost_low = c_crew(y) - 10000*(14-time(y,miles=200,cleanup_rate=5.0/7))`

We can't evaluate the `fine` function at `y` because it is not legal to test inequalities with the symbol `y`.

The symbolic cost expression `s_cost_low` matches \\(C_{tot}(n)\\) for \\(n\\) in the interval
\\([0,19)\\).

We must also understand the behaviour of \\(C_{tot}(n)\\) for \\(n>19\\). For that, we define

`s_cost_high = c_crew(y)`

So the symbolic expression `s_cost_high` matches \\(C_{tot}(n)\\) for \\(n \ge 19\\).

Now *sympy* permits us to symbolically differentiate these expressions:

```
ds_cost_low = sp.diff(s_cost_low,y)
ds_cost_high = sp.diff(s_cost_high,y)
```
We'll carry this out in the following cell:

```python
import sympy as sp

y = sp.Symbol('y')    # symbolic variable

s_cost_low  = c_crew(y) - 10000*(14-time(y,miles=200,cleanup_rate = 5.0/7))
s_cost_high = c_crew(y)

ds_cost_low  = sp.diff(s_cost_low,y)  # first derivative, for n<19
ds_cost_high = sp.diff(s_cost_high,y) # first derivative for n>19

dds_cost_low = sp.diff(ds_cost_low,y) # second derivative, for n<19

print(s_cost_low)
print()
print(ds_cost_low)
```

Remember that our goal is to find the minimum value(s) of the function \\(C_{tot}(n)\\) for \\(n\\) in the interval \\([0,\infty)\\).

We are going to argue a few things:
1. \\(C_{tot}\\) is increasing on the interval \\((19,\infty)\\)
2. \\(C_{tot}\\) has a unique critical point in the interval \\((0,19)\\), which is a local minimum


To argue 1., just need to show that \\(C'(n)\\) is positive on \\((19,\infty)\\).

But we've symbolically computed the derivative `ds_cost_high`.

And we can evaluate this symbolic expression at \\(y=19\\) by using
`ds_cost_high.subs([(y,19)])`. Here `subs` means *substitution*.

We see that \\(C'(19)\\) is positive:

```python
ds_cost_high.subs([(y,19)])
```

And this derivative is never zero. Indeed we can confirm using

`sp.solve(ds_cost_high,y)`

that the only solutions for \\(C_{crew}'(z) = 0\\) are complex numbers with non-zero imaginary part:

```python
sp.solve(ds_cost_high,y)
```

So since \\(C_{tot}'(n)\\) is continuous on \\((19,\infty)\\), we conclude that \\(C_{tot}'(n) > 0\\) on \\((19,\infty)\\) which confirms 1. above.

For 2., let's first look at a graph of the derivative \\(C'(n)\\)

```python

def diff(t):
    return ds_cost_low.subs([(y,t)])

x = np.linspace(7, 19, 200)

fig, ax = plt.subplots(figsize=(12,6))  
ax.plot(x,np.array([diff(z) for z in x]),label="(dC/dn)")
ax.legend()
ax.set_xlabel("#crews")
ax.set_ylabel("dollars / crew")
ax.axvline(x=11.28368,color="green",dashes=[1,4])
pass
```

The graph suggests that the equation \\(C'(n)=0\\) has only one solution in the interval \\((0,19)\\)

We can confirm this by again using the sympy symbolic solver to find the solutions to \\(C'(n) = 0\\), as follows:

```python
# find the critical points for our cost function
#

sp.solve(ds_cost_low,y)
```

So on the interval \\((0,19)\\) the cost function has only one critical point, at \\(n=11.28368\\).

In order to see that \\(n=11.28368\\) minimizes the cost function, we only need to compare the value of \\(C\\) at the endpoints with the value at \\(n=11.28368\\).

We compute:

```python
[c_tot(0),c_tot(11.28368),c_tot(19)]
```

This now confirms our [earlier observation](#minimal_cost_cell)  that \\(n=11\\) minimizes costs.




# Follow-up: Some general ideas behind modeling

1. Ask the question:
    * Here the question should be phrased correctly in mathematical terms; this will help make clear what must be found.
    * Make a list of all the variables and constants; include units as appropriate.
    * State all assumptions about these variables and constants; include equations and inequalities.
    * Check units to make sure things make sense.
    * State your objective in mathematical terms (i.e., "minimization problem" in the example above).
    * It may even be useful to make an educated guess at this point on what the answer should be. 
<!-- -->
2. Select the modeling approach.
    - Choose a general solution procedure to solve the mathematical problem (in our case first and second derivative tests).
    - This might be the most difficult part and to a large extent depends on just good experience. That’s our goal...to get some experience.
<!-- -->
3. Formulate the model.
    - Restate the question in terms of your model (in our example, what function are we taking the derivative of?).
    - You may need to relabel or redefine things to make it work. This is where the mathematical model and real physical model may start to differ...
<!-- -->
4. Solve the model.
    - Apply Step 2 to Step 3.
    - Use any useful technologies, such as computation if necessary, but consider the errors that they may introduce.
<!-- -->
5. Answer the question.
    - Rephrase the result of Step 4 in non-technical terms.
    - Goal is now to make your answer understandable to the person that posed it, keeping in mind that person may not be a mathematician.
    - Think about what the errors might be, or how realistic the answer actually is.
    - How did it compare to what expectations?

Of course this is very general and may be problem dependent, but it at least keeps us true to what
we are trying to do with modeling.
