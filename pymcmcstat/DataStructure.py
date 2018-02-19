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
        
#        self.xdata = [] # initialize list
#        self.ydata = [] # initialize list
#        self.n = [] # initialize list - number of data points
#        self.weight = [] # initialize list - weight of data set
#        self.user_defined_object = [] # user defined object
#        
               
        
#    def add_empty_data_batch(self, initial_batch_flag = False):
#        # Check if data already initialized
#        if initial_batch_flag is True:
#            self.batch = []
#            
##        self.batch.append(BasicDataStructure())
#        
##    def add_empty_data_batch(self, initial_batch_flag = 1):
##        # Check if data already initialized
##        if initial_batch_flag is None:
##            self.data = []
##            
##        self.data.append(DataStructure())
#        
#        self.batch.append({'xdata': None, 'ydata': None, 'n': None,
#                                'weight': None, 'user_defined_object': None})
        
    
    def add_data_set(self, x, y, n = None, weight = 1, user_defined_object = 0):
        # append new data set
        self.xdata.append(x)
        self.ydata.append(y)
        
        if n is None:
            if isinstance(y, list): # y is a list
                self.n.append(len(y))
            else:
                self.n.append(y.shape[0]) # assume y is a numpy array - nrows is n
        
        self.shape.append(y.shape)
        
        self.weight.append(weight)
        # add user defined objects option
        self.user_defined_object.append(user_defined_object)
        
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
        for ii in xrange(ndatabatches):
            nrows.append(dshapes[ii][0])
            if len(dshapes[0]) != 1:
                ncols.append(dshapes[ii][1])
        
        self.ndatasets = sum(ncols)
        
        return self.ndatasets
    
    def get_number_of_observations(self):
        n = np.sum(self.n)
        return np.array([n])