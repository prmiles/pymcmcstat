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
    '''
    Structure for storing data in MCMC object.
    The following random data sets will be referenced in examples for the
    different class methods:
    ::

        x1 = np.random.random_sample(size = (5, 1))
        y1 = np.random.random_sample(size = (5, 2))

        x2 = np.random.random_sample(size = (10, 1))
        y2 = np.random.random_sample(size = (10, 3))

    Attributes:
        * :meth:`~add_data_set`
        * :meth:`~get_number_of_batches`
        * :meth:`~get_number_of_data_sets`
        * :meth:`~get_number_of_observations`
    '''
    def __init__(self):
        self.xdata = []  # initialize list
        self.ydata = []  # initialize list
        self.n = []  # initialize list - number of data points
        self.shape = []  # shape of ydata - important if information stored as matrix
        self.weight = []  # initialize list - weight of data set
        self.user_defined_object = []  # user defined object

    def add_data_set(self, x, y, n=None, weight=1, user_defined_object=0):
        '''
        Add data set to MCMC object.

        This method must be called first before using any of the other methods
        within :class:`~DataStructure`.
        ::

            mcstat = MCMC()
            mcstat.data.add_data_set(x = x1, y = y1)
            mcstat.data.add_data_set(x = x2, y = y2)

        This yields the following variables in the data structure.

        * `xdata` (:py:class:`list`): List of numpy arrays
            - :code:`xdata[0] = x1, xdata[0].shape = (5,1)`
            - :code:`xdata[1] = x2, xdata[1].shape = (10,1)`
        * `ydata` (:py:class:`list`): List of numpy arrays
            - :code:`ydata[0] = y1, ydata[0].shape = (5,2)`
            - :code:`ydata[1] = y2, ydata[1].shape = (10,3)`
        * `n` (:py:class:`list`): List of integers. :code:`n = [5, 10]`
        * `shape` (:py:class:`list`): List of `y.shape`. :code:`shape = [(5,2),(10,3)]`
        * `weight` (:py:class:`list`): List of weights. :code:`weight = [1, 1]`
        * `user_defined_object` (:py:class:`list`): List of objects. :code:`user_defined_object = [0,0]`

        Args:
            * **x** (:class:`~numpy.ndarray`): Independent data.  Recommend input as column vectors.
            * **y** (:class:`~numpy.ndarray`): Dependent data.  Recommend input as column vectors.
            * **n** (:py:class:`list`): List of integers denoting number of data points.
            * **weight** (:py:class:`list`): Weight of each data set.
            * **user_defined_object** (`User Defined`): Any object can be stored in this variable.

        .. note::
            In general, it is recommended that user's format their data as a column
            vector.  So, if you have `nds` independent data points, `x` and `y` should be
            `[nds,1]` or `[nds,]` numpy arrays.  Note if a list is sent, the code will
            convert it to a numpy array.
        '''
        # check that x and y are numpy arrays
        x = self._convert_to_numpy_array(x)
        y = self._convert_to_numpy_array(y)
        # convert to 2d arrays (if applicable)
        x = self._convert_numpy_array_to_2d(x)
        y = self._convert_numpy_array_to_2d(y)
        # append new data set
        self.xdata.append(x)
        self.ydata.append(y)
        if n is None:
            if isinstance(y, list):  # y is a list
                self.n.append(len(y))
            elif isinstance(y, np.ndarray) and y.size == 1:
                self.n.append(y.size)
            else:  # should
                self.n.append(y.shape[0])  # assume y is a numpy array - nrows is n
        self.shape.append(y.shape)
        self.weight.append(weight)
        # add user defined objects option
        self.user_defined_object.append(user_defined_object)

    @classmethod
    def _convert_to_numpy_array(cls, xy):
        '''
        Convert variable to numpy array.

        Args:
            * **xy** (`Unknown`): Variable to be converted

        Returns:
            * **xy** (:class:`~numpy.ndarray`): Variable as numpy array
        '''
        if isinstance(xy, np.ndarray) is False:
            xy = np.array([xy])
        return xy

    @classmethod
    def _convert_numpy_array_to_2d(cls, xy):
        '''
        Convert numpy array to at least 2d numpy array.

        Args:
            * **xy** (:class:`~numpy.ndarray`): Variable to be checked/converted

        Returns:
            * **xy** (:class:`~numpy.ndarray`): Variable as at least 2d numpy array
        '''
        if xy.ndim != 2:  # numpy array is (xy.size,) -> Convert to (xy.size,1)
            xy = xy.reshape(xy.size, 1)
        return xy

    def get_number_of_batches(self):
        '''
        Get number of batches in data structure.  Essentially, each time you call
        the :meth:`~add_data_set` method you are adding another batch. It is
        also equivalent to say the number of batches is equal to the length of the
        list `ydata`.  For example,
        ::

            nb = mcstat.data.get_number_of_batches()

        should return :code:`nb = 2` because :code:`len(mcstat.data.ydata) = 2`.

        Returns:
            * **nbatch** (:py:class:`int`): Number of batches.
        '''
        self.nbatch = len(self.shape)
        return self.nbatch

    def get_number_of_data_sets(self):
        '''
        Get number of data sets in data structure.  A data set is strictly
        speaking defined as the total number of columns in each element of
        the `ydata` list. For example,
        ::

            nds = mcstat.data.get_number_of_data_sets()

        should return :code:`nds = 2 + 3 = 5` because the number of columns in `y1` is
        2 and the number of columns in `y2` is 3.

        Returns:
            * Number of columns in `ydata` (:py:class:`int`)
        '''
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
        '''
        Get number of observations in data structure.  An observation is essentially
        the total number of rows from each element of the `ydata` list. For example,
        ::

            nobs = mcstat.data.get_number_of_observations()

        should return :code:`nobs = 5 + 10 = 15` because the number of rows in `y1` is
        5 and the number of rows in `y2` is 10.

        Returns:
            * Number of rows in `ydata` (:class:`~numpy.ndarray`)
        '''
        n = np.sum(self.n)
        return np.array([n])
