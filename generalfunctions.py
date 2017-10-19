#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:50:08 2017

@author: prmiles
"""

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
        
def cholesky(A):
    """Performs a Cholesky decomposition of A, which must 
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L."""
    
    from math import sqrt
    
    n = len(A)

    # Create zero matrix for L
    L = [[0.0] * n for i in xrange(n)]

    # Perform the Cholesky decomposition
    for i in xrange(n):
        for k in xrange(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in xrange(k))
            
            if (i == k): # Diagonal elements
                # LaTeX: l_{kk} = \sqrt{ a_{kk} - \sum^{k-1}_{j=1} l^2_{kj}}
                L[i][k] = sqrt(A[i][i] - tmp_sum)
            else:
                # LaTeX: l_{ik} = \frac{1}{l_{kk}} \left( a_{ik} - \sum^{k-1}_{j=1} l_{ij} l_{kj} \right)
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L

def is_semi_pos_def_chol(x):
    import numpy as np
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.linalg.LinAlgError:
        return False

def print_mod(string, value, flag):
    if flag:
        print('{}{}'.format(string, value))
        
def display_dictionary(dictionary):
    print('Dictionary Contents:')
    attrs = vars(dictionary)
#            print('{} = {}\n' for item in attrs.items())
#            print ', '.join("%s: %s" % item for item in attrs.items())
    print ''.join('%s = %s\n' % item for item in attrs.items())