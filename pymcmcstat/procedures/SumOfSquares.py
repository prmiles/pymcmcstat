#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:21:48 2018

@author: prmiles
"""

# Import required packages
import numpy as np
import sys

class SumOfSquares:
    '''
    Sum-of-squares function evaluation.

    **Description:** Sum-of-squares (sos) class intended for used in MCMC simulator.  Each instance
    will contain the sos function.  If the user did not specify a sos-function,
    then the user supplied model function will be used in the default mcmc sos-function.

    Attributes:
        * :meth:`evaluate_sos_function`
        * :meth:`mcmc_sos_function`
    '''
    def __init__(self, model, data, parameters):
        # check if sos function and model function are defined
        if model.sos_function is None: #isempty(ssfun)
            if model.model_function is None: #isempty(modelfun)
                sys.exit('No ssfun or modelfun specified!')
            sos_style = 4
        else:
            sos_style = 1
        
        self.sos_function = model.sos_function
        self.sos_style = sos_style
        self.model_function = model.model_function
        self.parind = parameters._parind
        self.value = parameters._initial_value
        self.local = parameters._local
        self.data = data
        self.nbatch = model.nbatch
        
    def evaluate_sos_function(self, theta):
        '''
        Evaluate sum-of-squares function.

        Args:
            * **theta** (:class:`~numpy.ndarray`): Parameter values.

        Returns:
            * **ss** (:class:`~numpy.ndarray`): Sum-of-squares error(s)
        '''
        # evaluate sum-of-squares function
        self.value[self.parind] = theta
        if self.sos_style == 1:
            ss = self.sos_function(self.value, self.data)
        elif self.sos_style == 4:
            ss = self.mcmc_sos_function(self.value, self.data, self.nbatch, self.model_function)
        else:
            ss = self.sos_function(self.value, self.data, self.local)
        
        # make sure sos is a numpy array
        if not isinstance(ss, np.ndarray):
            ss = np.array([ss])
            
        return ss
     
    @classmethod
    def mcmc_sos_function(cls, theta, data, nbatch, model_function):
        '''
        Default sum-of-squares function.

        .. note::

            This method requires specifying a model function instead of a
            sum of squares function.  Not recommended for most applications.

        Basic formulation:

        .. math::

            SS_{q,i} = \sum [w_i(y^{data}_i-y^{model}_i)^2]

        where :math:`w_i` is the weight of a particular data set, and :math:`SS_{q,i}`
        is the sum-of-squares error for the `i`-th data set.

        Args:
            * **theta** (:class:`~numpy.ndarray`): Parameter values.

        Returns:
            * **ss** (:class:`~numpy.ndarray`): Sum-of-squares error(s)           
        '''
        # initialize
        ss = np.zeros(nbatch)

        for ibatch in range(nbatch):
                xdata = data.xdata[ibatch]
                ydata = data.ydata[ibatch]
                weight = data.weight[ibatch]
            
                # evaluate model
                ymodel = model_function(xdata, theta)
    
                # calculate sum-of-squares error
                ss[ibatch] += sum(weight*(ydata-ymodel)**2)
        
        return ss