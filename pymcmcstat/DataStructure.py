#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:03:37 2018

@author: prmiles
"""

# -------------------------
# Define data structure
# -------------------------
class DataStructure:
    def __init__(self):
        self.xdata = [] # initialize list
        self.ydata = [] # initialize list
        self.n = [] # initialize list - number of data points
        self.weight = [] # initialize list - weight of data set
        self.udobj = [] # user defined object
        
    def add_data_set(self, x, y, n = None, weight = 1, udobj = 0):
        # append new data set
        self.xdata.append(x)
        self.ydata.append(y)
        
        if n is None:
            if isinstance(y, list): # y is a list
                self.n.append(len(y))
            else:
                self.n.append(y.size) # assume y is a numpy array
        
        self.weight.append(weight)
        # add user defined objects option
        self.udobj.append(udobj)