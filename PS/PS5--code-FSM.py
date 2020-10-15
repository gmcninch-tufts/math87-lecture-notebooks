import numpy as np

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

#--------------------------------------------------------------------------------
from graphviz import Digraph

pop = Digraph("pop")
pop.attr(rankdir='LR')

p = list(range(5))
with pop.subgraph() as c:
#    c.attr(rank='same')
    for i in p:
        c.node(f"Age {i}")

for i in p:
    if i+1 in p:
        pop.edge(f"Age {i}",f"Age {i+1}",f"s{i}")
    if i != 0:
        pop.edge(f"Age {i}","Age 0",f"f{i}")
    
##--------------------------------------------------------------------------------
import numpy as np

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


def sbv(index,size):
    ## in this case, index should be in range(size) = [0,1,...,size-1].
    return np.array([1.0 if i == index else 0.0 for i in range(size)])

ones = np.ones(5)

def A(f,s):
    ## f and s should be lists each of length 5 with etnries between 0 and 1
    ## and s4 should be 0
    return np.array([np.array(f)]
                    +
                    [s[i]*sbv(i,5) for i in range(4)])
    
## for example,
## f1 = [.4,.25,.2,.15,0]
## s1 = [.4,.4,.4,.4,0]

## or 
## f1 = [.7,.6,.5,.25,0]
## s1 = [.6,.7,.5,1,0]

## You can compute powers of a square matrix as follows:
##
## A = A(f,s)
## np.linalg.matrix_power(A,5)

## And you can multiply the "all-ones" row vector ones by A using:
## ones @ A
