#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 06:24:12 2018

@author: prmiles
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy import pi,sin,cos
import sys
import math

def generate_subplot_grid(nparam = 2):
    '''
    Generate subplot grid.

    For example, if `nparam` = 2, then the subplot will have 2 rows and 1 column.

    Args:
        * **nparam** (:py:class:`int`): Number of parameters

    Returns:
        * **ns1** (:py:class:`int`): Number of rows in subplot
        * **ns2** (:py:class:`int`): Number of columns in subplot
    '''
    ns1 = math.ceil(math.sqrt(nparam))
    ns2 = round(math.sqrt(nparam))
    return ns1, ns2

def generate_names(nparam, names):
    '''
    Generate parameter name set.

    For example, if `nparam` = 4, then the generated names are::

        names = ['p_{0}', 'p_{1}', 'p_{2}', 'p_{3}']

    Args:
        * **nparam** (:py:class:`int`): Number of parameter names to generate
        * **names** (:py:class:`list`): Names of parameters provided by user

    Returns:
        * **names** (:py:class:`list`): List of strings - parameter names
    '''
    # Check if names defined
    if names == None:
        names = generate_default_names(nparam)

    # Check if enough names defined
    if len(names) != nparam:
        names = extend_names_to_match_nparam(names, nparam)
    return names

def setup_plot_features(nparam, names, figsizeinches):
    '''
    Setup plot features.

    Args:
        * **nparam** (:py:class:`int`): Number of parameters
        * **names** (:py:class:`list`): Names of parameters provided by user
        * **figsizeinches** (:py:class:`list`): [Width, Height]

    Returns:
        * **ns1** (:py:class:`int`): Number of rows in subplot
        * **ns2** (:py:class:`int`): Number of columns in subplot
        * **names** (:py:class:`list`): List of strings - parameter names
        * **figsizeiches** (:py:class:`list`): [Width, Height]
    '''
    ns1, ns2 = generate_subplot_grid(nparam = nparam)

    names = generate_names(nparam = nparam, names = names)
    
    if figsizeinches is None:
        figsizeinches = [5,4]
        
    return ns1, ns2, names, figsizeinches

def generate_default_names(nparam):
    '''
    Generate generic parameter name set.

    For example, if `nparam` = 4, then the generated names are::

        names = ['p_{0}', 'p_{1}', 'p_{2}', 'p_{3}']

    Args:
        * **nparam** (:py:class:`int`): Number of parameter names to generate

    Returns:
        * **names** (:py:class:`list`): List of strings - parameter names
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

    Args:
        * **names** (:py:class:`list`): Names of parameters provided by user
        * **nparam** (:py:class:`int`): Number of parameter names to generate

    Returns:
        * **names** (:py:class:`list`): List of strings - extended list of parameter names
    '''
    if names is None:
        names = []
        
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

    Args:
        * **x** (:class:`~numpy.ndarray`): Array of points
        * **npts** (:py:class:`int`): Number of points to use in generated grid

    Returns:
        * Uniformly spaced array of points with shape :code:`=(npts,1)`. (:class:`~numpy.ndarray`)
    '''
    xmin = min(x)
    xmax = max(x)
    xxrange = xmax-xmin
    if len(x) > 200:
        x_grid=np.linspace(xmin-0.08*xxrange,xmax+0.08*xxrange,npts)
    else:
        x_grid=np.linspace(np.mean(x)-4*np.std(x, ddof=1),np.mean(x)+4*np.std(x, ddof=1),npts)
    return x_grid.reshape(x_grid.shape[0],1) # returns 1d column vector

# --------------------------------------------
# see MASS 2nd ed page 181.
def iqrange(x):
    '''
    Interquantile range of each column of x.
    
    Args:
        * **x** (:class:`~numpy.ndarray`): Array of points.

    Returns:
        * (:class:`~numpy.ndarray`): Interquantile range - single element array, `q3 - q1`.
    '''
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
    
def gaussian_density_function(x, mu = 0, sigma2 = 1):
    '''
    Standard normal/Gaussian density function.
    
    Args:
        * **x** (:py:class:`float`): Value of which to calculate probability.
        * **mu** (:py:class:`float`): Mean of Gaussian distribution.
        * **sigma2** (:py:class:`float`): Variance of Gaussian distribution.

    Returns:
        * **y** (:py:class:`float`): Likelihood of `x`.
    '''
    y = 1/math.sqrt(2*math.pi*sigma2)*math.exp(-0.5*(x-mu)**2/sigma2)
    return y

def scale_bandwidth(x):
    '''
    Scale bandwidth of array.
    
    Args:
        * **x** (:class:`~numpy.ndarray`): Array of points - column of chain.

    Returns:
        * **s** (:class:`~numpy.ndarray`): Scaled bandwidth - single element array.
    '''
    n = len(x)
    if iqrange(x) <= 0:
        s = 1.06*np.array([np.std(x, ddof=1)*n**(-1/5)])
    else:
        s = 1.06*np.array([min(np.std(x, ddof=1),iqrange(x)/1.34)*n**(-1/5)])
    return s

# --------------------------------------------
def generate_ellipse(mu, cmat, ndp = 100):
    '''
    Generates points for a probability contour ellipse

    Args:
        * **mu** (:class:`~numpy.ndarray`): Mean values
        * **cmat** (:class:`~numpy.ndarray`): Covariance matrix
        * **npd** (:py:class:`int`): Number of points to generate

    Returns:
        * **x** (:class:`~numpy.ndarray`): x-points
        * **y** (:class:`~numpy.ndarray`): y-points
    '''
    
    # check shape of covariance matrix
    if cmat.shape != (2,2):
        sys.exit('covariance matrix must be 2x2')
    
    if check_symmetric(cmat) is not True:
        sys.exit('covariance matrix must be symmetric')
        
    
    # define t space
    t = np.linspace(0, 2*pi, ndp)
    
    pdflag, R = is_semi_pos_def_chol(cmat)
    if pdflag is False:
        sys.exit('covariance matrix must be positive definite')
    
    x = mu[0] + R[0,0]*cos(t)
    y = mu[1] + R[0,1]*cos(t) + R[1,1]*sin(t)
    
    return x, y

def check_symmetric(a, tol=1e-8):
    '''
    Check if array is symmetric by comparing with transpose.
    
    Args:
        * **a** (:class:`~numpy.ndarray`): Array to test.
        * **tol** (:py:class:`float`): Tolerance for testing equality.

    Returns:
        * (:py:class:`bool`): True -> symmetric, False -> not symmetric.
    '''
    return np.allclose(a, a.T, atol = tol)

def is_semi_pos_def_chol(x):
    '''
    Check if matrix is semi positive definite by calculating Cholesky decomposition.

    Args:
        * **x** (:class:`~numpy.ndarray`): Matrix to check

    Returns:
        * If matrix is `not` semi positive definite return :code:`False, None`
        * If matrix is semi positive definite return :code:`True` and the Upper triangular form of the Cholesky decomposition matrix.
    '''
    c = None
    try:
        c = np.linalg.cholesky(x)
        return True, c.transpose()
    except np.linalg.linalg.LinAlgError:
        return False, c
    
def append_to_nrow_ncol_based_on_shape(sh, nrow, ncol):
    '''
    Append to list based on shape of array

    Args:
        * **sh** (:py:class:`tuple`): Shape of array.
        * **nrow** (:py:class:`list`): List of number of rows
        * **ncol** (:py:class:`list`): List of number of columns

    Returns:
        * **nrow** (:py:class:`list`): List of number of rows
        * **ncol** (:py:class:`list`): List of number of columns
    '''
    if len(sh) == 1:
        nrow.append(sh[0])
        ncol.append(1)
    else:
        nrow.append(sh[0])
        ncol.append(sh[1])
    return nrow, ncol

# --------------------------------------------
def convert_flag_to_boolean(flag):
    '''
    Convert flag to boolean for backwards compatibility.

    Args:
        * **flag** (:py:class:`bool` or :py:class:`int`): Flag to specify something.

    Returns:
        * **flag** (:py:class:`bool`): Flag to converted to boolean.
    '''
    if flag is 'on':
        flag = True
    elif flag is 'off':
        flag = False
        
    return flag

# --------------------------------------------
def set_local_parameters(ii, local):
    '''
    Set local parameters based on tests.
    
    :Test 1:
        * `local == 0`
    :Test 2:
        * `local == ii`
    
    Args:
        * **ii** (:py:class:`int`): Index.
        * **local** (:class:`~numpy.ndarray`): Local flags.

    Returns:
        * **test** (:class:`~numpy.ndarray`): Array of Booleans indicated test results.
    '''
    # some parameters may only apply to certain batch sets
    test1 = local == 0
    test2 = local == ii
    test = test1 + test2
    return test.reshape(test.size,)

# --------------------------------------------
def empirical_quantiles(x, p = np.array([0.25, 0.5, 0.75])):
    '''
    Calculate empirical quantiles.

    Args:
        * **x** (:class:`~numpy.ndarray`): Observations from which to generate quantile.
        * **p** (:class:`~numpy.ndarray`): Quantile limits.

    Returns:
        * (:class:`~numpy.ndarray`): Interpolated quantiles.
    '''

    # extract number of rows/cols from np.array
    n = x.shape[0]
    # define vector valued interpolation function
    xpoints = range(n)
    interpfun = interp1d(xpoints, np.sort(x, 0), axis = 0)
    
    # evaluation points
    itpoints = (n-1)*p
    
    return interpfun(itpoints)

# --------------------------------------------    
def check_defaults(kwargs, defaults):
    '''
    Check if defaults are defined in kwargs
    
    Args:
        * **kwargs** (:py:class:`dict`): Keyword arguments.
        * **defaults** (:py:class:`dict`): Default settings.

    Returns:
        * **kwargs** (:py:class:`dict`): Updated keyword arguments with at least defaults set.
    '''
    for ii in defaults:
        if ii not in kwargs:
            kwargs[ii] = defaults[ii]
    return kwargs