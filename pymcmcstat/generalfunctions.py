#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:50:08 2017

@author: prmiles
"""

from __future__ import division
import math
import numpy as np
from scipy.interpolate import interp1d

def less_than_or_equal_to_zero(x):
    return (x<=0)

def replace_list_elements(x, testfunction, value):
    for ii in range(len(x)):
        if testfunction(x[ii]):
            x[ii] = value
    return x

def message(verbosity, level, printthis):
    if verbosity >= level:
        print(printthis)
        
def is_semi_pos_def_chol(x):
    c = None
    try:
        c = np.linalg.cholesky(x)
        return True, c.transpose()
    except np.linalg.linalg.LinAlgError:
        return False, c

def print_mod(string, value, flag):
    if flag:
        print('{}{}'.format(string, value))
        
#def display_dictionary(dictionary):
#    print('Dictionary Contents:')
#    attrs = vars(dictionary)
##            print('{} = {}\n' for item in attrs.items())
##            print ', '.join("%s: %s" % item for item in attrs.items())
#    print ''.join('%s = %s\n' % item for item in attrs.items())
    
def nordf(x, mu = 0, sigma2 = 1):
    # NORDF the standard normal (Gaussian) cumulative distribution.
    # NORPF(x,mu,sigma2) x quantile, mu mean, sigma2 variance
    # Marko Laine <Marko.Laine@Helsinki.FI>
    # $Revision: 1.5 $  $Date: 2007/12/04 08:57:00 $
    # adapted for Python by Paul Miles, November 8, 2017
    return 0.5 + 0.5*math.erf((x-mu)/math.sqrt(sigma2)/math.sqrt(2))

def empirical_quantiles(x, p = np.array([0.25, 0.5, 0.75])):
    """
    function y=plims(x,p)
    %PLIMS Empirical quantiles
    % plims(x,p)  calculates p quantiles from columns of x
    % Marko Laine <Marko.Laine@Helsinki.FI>
    % $Revision: 1.4 $  $Date: 2007/05/21 11:19:12 $
    Adapted for Python by Paul Miles on 2017/11/08
    """

    # extract number of rows/cols from np.array
    n, m = x.shape 
    
    # define vector valued interpolation function
    xpoints = range(n)
    interpfun = interp1d(xpoints, np.sort(x, 0), axis = 0)
    
    # evaluation points
    itpoints = (n-1)*p + 1    
    
    return interpfun(itpoints)