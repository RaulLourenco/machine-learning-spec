import numpy as np

#for loop way
def sq_dist(a,b):
    n = a.shape[0]
    d = 0.
    for i in range(n): 
        d += (a[i] - b[i])**2
    return d

#numpy aux to do the same
def sq_dist(a,b):
    return np.sum(np.square(a - b))