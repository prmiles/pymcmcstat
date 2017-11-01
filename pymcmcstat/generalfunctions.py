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
        
def is_semi_pos_def_chol(x):
    import numpy as np
    c = None
    try:
        c = np.linalg.cholesky(x)
        return True, c.transpose()
    except np.linalg.linalg.LinAlgError:
        return False, c

def print_mod(string, value, flag):
    if flag:
        print('{}{}'.format(string, value))
        
def display_dictionary(dictionary):
    print('Dictionary Contents:')
    attrs = vars(dictionary)
#            print('{} = {}\n' for item in attrs.items())
#            print ', '.join("%s: %s" % item for item in attrs.items())
    print ''.join('%s = %s\n' % item for item in attrs.items())