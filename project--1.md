---
{{{abstract}}}
<!-- #region -->

In this problem, you are asked to first to minimize the shipping costs
of a supply chain system between several cities, and then maximize the
overall profit, taking into account various situations.  Consider
yourself to be the logistics manager for a supply-chain company that
makes and sells rubber ducks. You have 3 main warehouses in Santa Fe,
El Paso, and Tampa Bay. At each ware- house, you are given a certain
number of rubber ducks that must be shipped to your stores in various
cities across the US. The number of supplies (in units of ducks) for
each warehouse is listed here:

| Santa Fe | El Paso | Tampa Bay|
|---------:|--------:|---------:|
|      700 |     200 |      700 |

You have 5 stores located across the US that will sell these ducks to
your customers. The demands at each store are as follows (again in
units of ducks):

| Chicago | LA | NY | Houston | Atlanta |
|--------:|---:|---:|--------:|--------:|
|     200 | 200| 250|      300|      150|


In order to ship the rubber ducks, to each of these cities, you use an
air-shipping service that charges different prices between different
cities depending on how much you ship. Some routes are not
available. The following grid indicates the cost (in dollars) to ship
one duck between a warehouse and a store. Note, these routes are
one-way:

|           | Chicago | LA | NY | Houston | Atlanta |
|----------:|--------:|---:|---:|--------:|--------:|
|  Santa Fe |       6 |  3 |  - |       3 |       7 |
|   El Paso |       - |  7 |  - |       2 |       5 |
| Tampa Bay |       - |  - |  7 |       6 |       4 |


Additionally, Houston and Atlanta are hubs that, in addition to their
own demands, can trans- ship material to other customers. Those routes
are indicated here:

|         | Chicago | LA | NY | Houston | Atlanta |
|--------:|--------:|---:|---:|--------:|--------:|
| Houston |       4 |  5 |  6 |       - |       2 |
| Atlanta |       4 |  - |  5 |       2 |       - |


Finally, we should note that shipping on each route is restricted to a
maximum of 200 units.  The problem is to determine an optimum shipping
plan that minimizes the total cost of shipping while meeting all
customer demands with available supplies. Your task is to formulate
and solve (in matlab) a linear program to solve this problem, subject
to the constraints explained above. 

For simplicity, it’s ok if you get partial ducks...but only in extreme
situations... For this midterm, you must:

1. Draw a clearly labeled network-flow model for the linear
program. You must explain all the constraints that you have included
and why you have included them. I strongly encourage you to include a
node for the source of ducks and a node for the customers (a sink)
even though nothing really gets “shipped” to these nodes.

2. Use your network-flow model to write out the linear program. Be
sure to use descriptive variable names.

3. Input your model into matlab and use the linprog command to solve
it. Be sure to include a “legend” that explains the ordering of your
descriptive variable names in the solution vector used by
matlab. Include a printout of your setup and call to linprog, as well
as the output.

4. Next, consider the following variants. Assume that shipping workers
in LA are unhappy and considering to strike. They demand that all
shipping costs to LA be doubled, otherwise they strike and the maximum
number of supplies that can be shipped on each route is cut in half
(i.e., from 200 to 100). Model both scenarios and see which one
increases the cost more.

5. Test the same scenarios on the hub city of Houston. Is the result
more drastic? Which city (LA or Houston) would cause the most problems
for us if a work stoppage occurred?

6. Finally, (in the non-strike scenario), consider the value of the
goods being made and sold.  The following table shows the profit made
at each city from selling 1 rubber duck. Note that in the warehouse
cities, you are making the goods, which is a cost instead of a
profit. Now, use your same linear program to maximize the total
profit, taking into account the shipping costs.


| Santa fe | El Paso | Tampa Bay | Chicago | NY | Houston | Atlanta | LA |
|---------:|--------:|----------:|--------:|---:|--------:|--------:|---:|
|       -8 |      -5 |       -10 |      15 | 25 |      10 |      10 | 20 |

7. Complete a project report that is typed (aside from the
network-flow model and resulting linear program, which may be
hand-drawn and hand-written, respectively) and written in the style of
a lab report and not that of a problem set. This should be uploaded to
Canvas and submitted as one pdf file. You may attach your code
separately, but you must include a README file to explain how to run
it. Fully explain how you formulated your linear programs (including
all choices of nodes and edges). Write a summary of your solutions in
economic terms, identifying the optimal shipping routes and how to
deal with unexpected crises. See the guidelines for the final project
to get an idea of the expected formatting. Of course, the guidelines
for this midterm are not as rigorous (i.e., no abstract or references
are needed).
