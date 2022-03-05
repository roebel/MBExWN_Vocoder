# coding: utf-8
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2014 - 2022 IRCAM, Roebel
#

"""
support functions for converting positive real values from linear to db and inverse
"""

from __future__ import division, absolute_import

import numpy as np

def lin2db(vec, l_no_abs = False, minthresh=None) :
    if l_no_abs:
        if minthresh is None:
            return 20.*np.log10(vec)
        else:
            return 20.*np.log10(np.fmax(vec, minthresh))
        
    if minthresh is None:
        return 20.*np.log10(np.abs(vec))
    return 20.*np.log10(np.fmax(np.abs(vec), minthresh))

def db2lin(vec) :
    return 10**(vec/20.)
