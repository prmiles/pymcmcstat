#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:06:51 2018

@author: prmiles
"""

# import required packages
import numpy as np
import sys
#import warnings

class ModelSettings:
    '''
    MCMC Model Settings

    Attributes:
        * :meth:`~define_model_settings`
        * :meth:`~display_model_settings`
    '''
    def __init__(self):
        # Initialize all variables to default values
        self.description = 'Model Settings'

    def define_model_settings(self, sos_function = None, prior_function = None, prior_type = 1,
                 prior_update_function = None, prior_pars = None, model_function = None,
                 sigma2 = None, N = None, S20 = np.nan, N0 = None, nbatch = None):
        '''
        Define model settings.

        Args:
            * **sos_function**: Handle for sum-of-squares function
            * **prior_function**: Handle for prior function
            * **prior_type**: Pending...
            * **prior_update_function**: Pending...
            * **prior_pars**: Pending...
            * **model_function**: Handle for model function (needed if :code:`sos_function` not specified)
            * **sigma2** (:py:class:`float`): List of initial error observations.
            * **N** (:py:class:`int`): Number of observations - see :class:`~.DataStructure`.
            * **S20** (:py:class:`float`): List of scaling parameter in observation error estimate.
            * **N0** (:py:class:`float`): List of scaling parameter in observation error estimate.
            * **nbatch** (:py:class:`int`): Number of batch data sets - see :meth:`~.get_number_of_batches`.

        .. note:: Variables :code:`sigma2, N, S20, N0`, and :code:`nbatch` converted to :class:`~numpy.ndarray` for subsequent processing.
        '''
    
        self.sos_function = sos_function
        self.prior_function = prior_function
        self.prior_type = prior_type
        self.prior_update_function = prior_update_function
        self.prior_pars = prior_pars
        self.model_function = model_function
        
        # check value of sigma2 - initial error variance
        self.sigma2 = self._array_type(sigma2)
        
        # check value of N - total number of observations
        self.N = self._array_type(N)
        
        # check value of N0 - prior accuracy for S20
        self.N0 = self._array_type(N0)
        
        # check nbatch - number of data sets
        self.nbatch = self._array_type(nbatch)
            
        # S20 - prior for sigma2
        self.S20 = self._array_type(S20)
    
    @classmethod
    def _array_type(cls, x):
        # All settings in this class should be converted to numpy ndarray
        if x is None:
            return None
        else:
            if isinstance(x, int): # scalar -> ndarray[scalar]
                return np.array([np.array(x)])
            elif isinstance(x, float): # scalar -> ndarray[scalar]
                return np.array([np.array(x)])
            elif isinstance(x, list): # [...] -> ndarray[...]
                return np.array(x)
            elif isinstance(x, np.ndarray):
                return x
            else:
                sys.exit('Unknown data type - Please use int, ndarray, or list')
    
    def _check_dependent_model_settings(self, data, options):
        '''
        Check dependent parameters.

        Args:
            * **data** (:class:`~.DataStructure`): MCMC data structure
            * **options** (:class:`.SimulationOptions`): MCMC simulation options
        '''
        if self.nbatch is None:
            self.nbatch = data.get_number_of_batches()
            
        if self.N is not None:
            self.N = self._check_number_of_observations(udN = self.N, dsN = self._array_type(data.n))
        else:
#            self.N = data.get_number_of_observations()
            self.N = data.n
        
        self.Nshape = data.shape
        
        # This is for backward compatibility
        # if sigma2 given then default N0=1, else default N0=0
        if self.N0 is None:
            if self.sigma2 is None:
                self.sigma2 = np.ones([1])
                self.N0 = np.zeros([1])
            else:
                self.N0 = np.ones([1])
            
        # set default value for sigma2
        # default for sigma2 is S20 or 1
        if self.sigma2 is None:
            if not(np.isnan(self.S20)).any:
                self.sigma2 = self.S20
            else:
                self.sigma2 = np.ones(self.nbatch)
        
        if np.isnan(self.S20).any():
            self.S20 = self.sigma2  # prior parameters for the error variance
      
    @classmethod
    def _check_number_of_observations(cls, udN, dsN):
        # check if user defined N matches data structure
        if np.array_equal(udN, dsN):
            N = dsN
        elif dsN.size > udN.size and udN.size == 1:
            if np.all(dsN == udN[0]):
                N = dsN
            else:
                sys.exit('User defined N = {}.  Estimate based on data structure is N = {}.  Possible error?'.format(udN, dsN))
        elif udN.size > dsN.size and dsN.size == 1:
            if np.all(dsN[0] == udN):
                N = udN
            else:
                sys.exit('User defined N = {}.  Estimate based on data structure is N = {}.  Possible error?'.format(udN, dsN))
        else:
            sys.exit('User defined N = {}.  Estimate based on data structure is N = {}.  Possible error?'.format(udN, dsN))
                
        return N
    
    def _check_dependent_model_settings_wrt_nsos(self, nsos):
        '''
        Check dependent model settings with respect to number of sum-of-square elements.

        A different observation error can be set up for different data sets, so these
        settings must be updated with respect to the length of the output from the
        sum-of-squares function evaluation.

        Args:
            * **nsos** (:py:class:`int`): Length of output from sum-of-squares function
        '''
        # in matlab version, ny = length(sos) where sos is the output from the sos evaluation
        self.S20 = self.__check_size_of_setting_wrt_nsos(self.S20, nsos)
        self.N0 = self.__check_size_of_setting_wrt_nsos(self.N0, nsos)
        self.sigma2 = self.__check_size_of_setting_wrt_nsos(self.sigma2, nsos)
        self.N = self.__check_size_of_observations_wrt_nsos(self.N, nsos, self.Nshape)
#        if len(self.N) == 1:
#            self.N = np.ones(nsos)*self.N
            
#        if len(self.N) == nsos + 1:
#            self.N = self.N[1:] # remove first column
            
        self.nsos = nsos
        
    @classmethod
    def __check_size_of_setting_wrt_nsos(cls, x, nsos):
        '''
        Check size of setting with respect number of observation errors.

        Args:
            * **x** (:class:`~numpy.ndarray`): Array to be checked
            * **nsos** (:py:class:`int`): Length of output from sum-of-squares function

        Returns:
            * **x** (:class:`~numpy.ndarray`): Array returned with shape = :code:`(nsos,1)`

        Raises:
            * Dimension mismatch if :code:`len(x) > nsos` or :code:`len(x) < nsos and len(x) != 1`.
        '''
        if len(x) == 1:
            x = np.ones(nsos)*x
        elif len(x) > nsos: # more x elements than sos elements
            sys.exit('SOS function returns nsos = {} elements.  Expect same number of elements in S20, N0, sigma2'.format(nsos))
        elif len(x) < nsos and len(x) != 1:
            sys.exit('SOS function returns nsos = {} elements.  Length of S20, N0, or sigma2 is {}.  These settings must have one element or nsos elements.'.format(nsos, len(x)))
        return x
     
    @classmethod
    def __check_size_of_observations_wrt_nsos(cls, N, nsos, Nshape):
        if len(N) == 1:
            N = np.ones(nsos)*N
        elif len(N) < nsos and len(N) > 1:
            sys.exit('Unclear data structure!  SOS function return nsos = {} elements.  len(N) = {}.'.format(nsos, len(N)))
        elif nsos == 1 and len(N) > 1:
            N = np.array([np.sum(N)])
            
        return N
            
    def display_model_settings(self, print_these = None):
        '''
        Display subset of the simulation options.

        Args:
            * **print_these** (:py:class:`list`): List of strings corresponding to keywords.  Default below.

        ::

            print_these = ['sos_function', 'model_function', 'sigma2', 'N', 'N0', 'S20', 'nsos', 'nbatch']
        '''
        if print_these is None:
            print_these = ['sos_function', 'model_function', 'sigma2', 'N', 'N0', 'S20', 'nsos', 'nbatch']
            
        print('model settings:')
        for ptii in print_these:
            print('\t{} = {}'.format(ptii, getattr(self, ptii)))
            
        return print_these