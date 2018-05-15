#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:03:37 2018

@author: prmiles
"""
# import required packages
import numpy as np

# -------------------------
# Define data structure
# -------------------------
class DataStructure:
    def __init__(self):
        self.xdata = [] # initialize list
        self.ydata = [] # initialize list
        self.n = [] # initialize list - number of data points
        self.shape = [] # shape of ydata - important if information stored as matrix
        self.weight = [] # initialize list - weight of data set
        self.user_defined_object = [] # user defined object
        
    
    def add_data_set(self, x, y, n = None, weight = 1, user_defined_object = 0):
        # in general, it is recommended that user's format their data as a column
        # vector.  So, if you have nds independent data points, x and y should be
        # [nds,1] or [nds,] numpy arrays.  Note if a list is sent, the code will 
        # convert it to a numpy array.
        
        # check that x and y are numpy arrays
        x = self.__convert_to_numpy_array(x)
        y = self.__convert_to_numpy_array(y)
        
        # convert to 2d arrays (if applicable)
        x = self.__convert_numpy_array_to_2d(x)
        y = self.__convert_numpy_array_to_2d(y)
        
        # append new data set
        self.xdata.append(x)
        self.ydata.append(y)
        
        if n is None:
            if isinstance(y, list): # y is a list
                self.n.append(len(y))
            elif isinstance(y, np.ndarray) and y.size == 1:
                self.n.append(y.size)
            else: # should 
                self.n.append(y.shape[0]) # assume y is a numpy array - nrows is n
        
        self.shape.append(y.shape)
        
        self.weight.append(weight)
        # add user defined objects option
        self.user_defined_object.append(user_defined_object)
        
    def __convert_to_numpy_array(self, xy):
        if isinstance(xy, np.ndarray) is False:
            xy = np.array([xy])
            
        return xy
    
    def __convert_numpy_array_to_2d(self, xy):
        if xy.ndim != 2: # numpy array is (xy.size,) -> Convert to (xy.size,1)
            xy = xy.reshape(xy.size,1)
            
        return xy
        
    def check_data_type(self, xy):
        
        if isinstance(xy, np.ndarray):
            str('all is well')
            
        return True
    
    def get_number_of_batches(self):
        self.nbatch = len(self.shape)
        return self.nbatch
    
    def get_number_of_data_sets(self):
        dshapes = self.shape
        ndatabatches = len(dshapes)
        nrows = []
        ncols = []
        for ii in range(ndatabatches):
            nrows.append(dshapes[ii][0])
            if len(dshapes[ii]) != 1:
                ncols.append(dshapes[ii][1])
        
        self.ndatasets = sum(ncols)
        
        return self.ndatasets
    
    def get_number_of_observations(self):
        n = np.sum(self.n)
        return np.array([n])