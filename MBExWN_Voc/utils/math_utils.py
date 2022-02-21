from __future__ import division
import numpy as np

def nextpow2(n):
    '''
    return next power of 2 of such that 2**nextpow2(n) >= n
    '''
    return np.int32(np.ceil(np.log2(n)))

def nextpow2_val(n):
    '''
    return integer v being a power of 2 with v >= n
    '''
    v = 2
    while v < n: v = v * 2
    return v
    

