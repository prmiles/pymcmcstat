#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 06:24:12 2018

@author: prmiles
"""

import numpy as np
import math

def generate_default_names(nparam):
    '''
    Generate generic parameter name set.
    For example, if `nparam` = 4, then the generated names are::
    
        names = ['p_{0}', 'p_{1}', 'p_{2}', 'p_{3}']
    
    :param nparam: Number of parameter names to generate
    :type nparam: :py:class:`int`
    
    :returns: Parameter names
    :rtype: :py:class:`list`
    '''
    names = []
    for ii in range(nparam):
        names.append(str('$p_{{{}}}$'.format(ii)))
    return names

def extend_names_to_match_nparam(names, nparam):
    '''
    Append names to list using default convention
    until length of names matches number of parameters.
    For example, if `names = ['name_1', 'name_2']` and `nparam = 4`, then
    two additional names will be appended to the `names` list.
    E.g.,::
        
        names = ['name_1', 'name_2', 'p_{2}', 'p_{3}']
    
    :param names: Names of parameters provided by user
    :type names: :py:class:`list`
    :param nparam: Number of parameters requiring a name
    :type nparam: :py:class:`int`
    
    :returns: Extended list of parameter names
    :rtype: :py:class:`list`
    '''
    n0 = len(names)
    for ii in range(n0,nparam):
        names.append(str('$p_{{{}}}$'.format(ii)))
    return names

# --------------------------------------------    
def make_x_grid(x, npts = 100):
    '''
    Generate x grid based on extrema.
    
    1. If `len(x) > 200`, then generates grid based on difference
    between the max and min values in the array.
    
    2. Otherwise, the grid is defined with respect to the array
    mean plus or minus four standard deviations.
    
    :param x: Array of points
    :type x: :class:`~numpy.ndarray`
    :param npts: Number of points to use in generated grid
    :type npts: :py:class:`int`
    
    :returns: Uniformly spaced array of points with shape (npts,1)
    :rtype: :class:`~numpy.ndarray`
    '''
    xmin = min(x)
    xmax = max(x)
    xxrange = xmax-xmin
    if len(x) > 200:
        x_grid=np.linspace(xmin-0.08*xxrange,xmax+0.08*xxrange,npts)
    else:
        x_grid=np.linspace(np.mean(x)-4*np.std(x, ddof=1),np.mean(x)+4*np.std(x, ddof=1),npts)
    return x_grid.reshape(x_grid.shape[0],1) # returns 1d column vector

#if iqrange(x)<=0
#  s=1.06*std(x)*nx^(-1/5);
#else
#  s=1.06*min(std(x),iqrange(x)/1.34)*nx^(-1/5);
#end
#

# --------------------------------------------    
"""see MASS 2nd ed page 181."""
def __iqrange(x):
    nr, nc = x.shape
    if nr == 1: # make sure it is a column vector
        x = x.reshape(nc,nr)
        nr = nc
        nc = 1
    
    # sort
    x.sort()
    
    i1 = math.floor((nr + 1)/4)
    i3 = math.floor(3/4*(nr+1))
    f1 = (nr+1)/4-i1
    f3 = 3/4*(nr+1)-i3
    q1 = (1-f1)*x[int(i1),:] + f1*x[int(i1)+1,:]
    q3 = (1-f3)*x[int(i3),:] + f3*x[int(i3)+1,:]
    return q3-q1
    
def __gaussian_density_function(x, mu, sigma2):
    y = 1/math.sqrt(2*math.pi*sigma2)*math.exp(-0.5*(x-mu)**2/sigma2)
    return y

def __scale_bandwidth(x):
    n = len(x)
    if __iqrange(x) <=0:
        s = 1.06*np.std(x, ddof=1)*n**(-1/5)
    else:
        s = 1.06*min(np.std(x, ddof=1),__iqrange(x)/1.34)*n**(-1/5)
    return s