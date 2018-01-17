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
                self.n.append(y.size) # assume y is a numpy array
        
        self.weight.append(weight)
        # add user defined objects option
        self.user_defined_object.append(user_defined_object)
        
    def check_data_type(self, xy):
        
        if isinstance(xy, np.ndarray):
            str('all is well')
            
        return True
    
    def get_number_of_batches(self):
        return np.array([len(self.n)])
    
    def get_number_of_observations(self):
        n = np.sum(self.n)
        return np.array([n])