#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:13:03 2018

@author: prmiles
"""
# import required packages
import numpy as np
import math
import sys
from ..utilities.general import message
# --------------------------
class ModelParameters:
    '''
    MCMC Model Parameters.

    Example:
    ::

        mcstat = MCMC()

        mcstat.parameters.add_model_parameter(name = 'm', theta0 = 1., minimum = -10, maximum = 10)
        mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100)
        mcstat.parameters.display_model_parameter_settings()

    This will display to screen:
    ::

        Sampling these parameters:
        name         start [   min,    max] N(  mu, sigma^2)
        m         :   1.00 [-10.00,  10.00] N(0.00, inf)
        b         :  -5.00 [-10.00, 100.00] N(0.00, inf)

    Attributes:
        * :meth:`~add_model_parameter`
        * :meth:`~display_parameter_settings`
    '''
    def __init__(self):
        self.parameters = [] # initialize list
        self.description = 'MCMC model parameters'
    # --------------------------
    def add_model_parameter(self, name = None, theta0 = None, minimum = -np.inf,
                      maximum = np.inf, prior_mu = np.zeros([1]), prior_sigma = np.inf,
                      sample = True, local = 0):
        '''
        Add model parameter to MCMC simulation.

        Args:
            * name (:py:class:`str`): Parameter name
            * theta0 (:py:class:`float`): Initial value
            * minimum (:py:class:`float`): Lower parameter bound
            * maximum (:py:class:`float`): Upper parameter bound
            * prior_mu (:py:class:`float`): Mean value of prior distribution
            * prior_sigma (:py:class:`float`): Standard deviation of prior distribution
            * sample (:py:class:`bool`): Flag to turn sampling on (True) or off (False)
            * local (:py:class:`int`): Local flag - still testing.

        The default prior is a uniform distribution from minimum to maximum parameter value.
        '''
        if name is None:
            name = generate_default_name(len(self.parameters))

        if theta0 is None:
            theta0 = 1.0

        # append dictionary element
        self.parameters.append({'name': name, 'theta0': theta0, 'minimum': minimum,
                                'maximum': maximum, 'prior_mu': prior_mu, 'prior_sigma': prior_sigma,
                                'sample': sample, 'local': local})
    # --------------------------
    def _openparameterstructure(self, nbatch):
        
        # unpack input object
        parameters = self.parameters
        npar = len(parameters)
        
        # initialize arrays - as lists and numpy arrays (improved functionality)
        self._names = []
        self._initial_value = np.zeros(npar)
        self._value = np.zeros(npar)
        self._parind = np.ones(npar, dtype = int)
        self._local = np.zeros(npar)
        self._upper_limits = np.ones(npar)*np.inf
        self._lower_limits = -np.ones(npar)*np.inf
        self._thetamu = np.zeros(npar)
        self._thetasigma = np.ones(npar)*np.inf
                
        # scan for local variables
        # ***************************
        # UPDATING THIS SECTION
#        self._local = self.scan_for_local_variables(nbatch = nbatch, parameters = parameters)
        ii = 0
        for kk in range(npar):
            if parameters[kk]['sample'] == 0:
                if parameters[kk]['local'] != 0:
                    self._local[ii:(ii+nbatch-1)] = range(0,nbatch)
                    npar = npar + nbatch - 1
                    ii = ii + nbatch - 1

            ii += 1 # update counter
        # ***************************
            
        ii = 0
        for kk in range(npar):
            if self._local[ii] == 0:
                self._names.append(parameters[kk]['name'])
                self._initial_value[ii] = parameters[kk]['theta0']
                self._value[ii] = parameters[kk]['theta0']
                # default values defined in "Parameters" class in classes.py
                # define lower limits
                self._lower_limits[ii] = parameters[kk]['minimum']
                # define upper limits
                self._upper_limits[ii] = parameters[kk]['maximum']
                # define prior mean
                self._thetamu[ii] = self.setup_prior_mu(mu = parameters[kk]['prior_mu'], value = self._value[ii])
                self._thetasigma[ii] = self.setup_prior_sigma(sigma = parameters[kk]['prior_sigma'])
                # turn sampling on/off
                self._parind[ii] = parameters[kk]['sample']
            ii += 1 # update counter
            
        # make parind list of nonzero elements
        self._parind = np.flatnonzero(self._parind)
        
        self.npar = len(self._parind) # append number of parameters to structure
    
    @classmethod
    def scan_for_local_variables(cls, nbatch, parameters):
        '''
        Scan for local variables
        
        Args:
            * **nbatch** (:py:class:`int`): Number of data batches
            * **parameters** (:py:class:`list`): List of model parameters.
        
        Returns:
            * **local** (:class:`~numpy.ndarray`): Array with local flag indices.
        '''
        local = np.array([], dtype = int)
        for kk in range(len(parameters)):
            if parameters[kk]['sample'] is True:
                if parameters[kk]['local'] != 0:
                    local = np.concatenate((local, range(1,nbatch+1)))
                else:
                    local = np.concatenate((local, np.zeros([1])))
        
        return local
  
    # --------------------------
    @classmethod
    def setup_prior_mu(cls, mu, value):
        '''
        Setup prior mean.
        
        Args:
            * **mu** (:py:class:`float`): defined mean
            * **value** (:py:class:`float`): default value
            
        Returns:
            * Prior mean
        '''
        if np.isnan(mu):
            return value
        else:
            return mu
        
    # --------------------------
    @classmethod
    def setup_prior_sigma(cls, sigma):
        '''
        Setup prior variance.
        
        Args:
            * **sigma** (:py:class:`float`): defined variance
            
        Returns:
            * Prior mean
        '''
        if sigma == 0:
            return np.inf
        else:
            return sigma
        
    # --------------------------        
    def _results_to_params(self, results, use_local = 1):
    
        # unpack results dictionary
        parind = results['parind']
        names = results['names']
        local = results['local']
        theta = results['theta']

        for ii, parii in enumerate(parind):
            if use_local == 1 and local[parii] == 1:
                name = names[ii] # unclear usage
            else:
                name = names[ii]
    
            for kk in range(len(self.parameters)):
                if name == self.parameters[kk]['name']:
                    # change NaN prior mu (=use initial) to the original initial value
                    if np.isnan(self.parameters[kk]['prior_mu']):
                        self.parameters[kk]['prior_mu'] = self.parameters[kk]['theta0']
                        
                    # only change if parind = 1 in params (1 is the default)
                    if self.parameters[kk]['sample'] == 1 or self.parameters[kk]['sample'] is None:
                        self.parameters[kk]['theta0'] = theta[parii]
                        
    # --------------------------
    def _check_initial_values_wrt_parameter_limits(self):
        # check initial parameter values are inside range
        if (self._initial_value[np.ix_(self._parind)] < self._lower_limits[np.ix_(self._parind)]).any() or (self._initial_value[np.ix_(self._parind)] > self._upper_limits[np.ix_(self._parind)]).any():
            # proposed value outside parameter limits
            sys.exit('Proposed value outside parameter limits - select new initial parameter values')
        else:
             return True
    # --------------------------     
    def _check_prior_sigma(self, verbosity):
        message(verbosity, 2, 'If prior variance <= 0, setting to Inf\n')
        self._thetasigma = replace_list_elements(self._thetasigma, less_than_or_equal_to_zero, float('Inf'))
    # --------------------------
    def display_parameter_settings(self, verbosity = None, noadaptind = None):
        '''
        Display parameter settings

        Args:
            * **verbosity** (:py:class:`int`): Verbosity of display output. :code:`0`
            * **noadaptind** (:py:class:`int`): Indices not to be adapted in covariance matrix. :code:`[]`
        '''
        parind = self._parind
        names = self._names
        value = self._initial_value
        lower_limits = self._lower_limits
        upper_limits = self._upper_limits
        theta_mu = self._thetamu
        theta_sigma = self._thetasigma
        
        verbosity = check_verbosity(verbosity)
        noadaptind = check_noadaptind(noadaptind)
            
        if noadaptind is None:
            noadaptind = []
        
        if verbosity > 0:
            print('\nSampling these parameters:')
            print('{:>10s} {:>10s} [{:>9s}, {:>9s}] N({:>9s}, {:>9s})'.format('name',
                  'start', 'min', 'max', 'mu', 'sigma^2'))
            nprint = len(parind)
            for ii in range(nprint):
                name = str('{:>10s}'.format(names[parind[ii]]))
                valuestr = format_number_to_str(value[parind[ii]])
                lowstr = format_number_to_str(lower_limits[parind[ii]])
                uppstr = format_number_to_str(upper_limits[parind[ii]])
                mustr = format_number_to_str(theta_mu[parind[ii]])
                sigstr = format_number_to_str(theta_sigma[parind[ii]])
                
                st = noadapt_display_setting(ii, noadaptind)
                h2 = prior_display_setting(x = theta_sigma[parind[ii]])
                
                print('{:s}: {:s} [{:s}, {:s}] N({:s},{:s}{:s}){:s}'.format(name, valuestr, lowstr, uppstr, mustr, sigstr, h2, st))

# --------------------------
def replace_list_elements(x, testfunction, value):
    '''
    Replace list elements based on results from testfunction.

    Args:
        * **x** (:py:class:`list`): List of numbers to be tested
        * **testfunction** (:py:func:`testfunction`): Test function
        * **value** (:py:class:`float`): Value to assign if test function return True

    Returns:
        * **x** (:py:class:`list`): Updated list
    '''
    for ii, xii in enumerate(x):
        if testfunction(xii):
            x[ii] = value
    return x
# --------------------------
def generate_default_name(nparam):
    '''
    Generate generic parameter name.
    For example, if :code:`nparam = 4`, then the generated name is::

        names = 'p_{3}'

    Args:
        * **nparam** (:py:class:`int`): Number of parameter names to generate

    Returns:
        * **name** (:py:class:`str`): Name based on size of parameter list
    '''
    return (str('$p_{{{}}}$'.format(nparam)))
# --------------------------
def check_verbosity(verbosity):
    '''
    Check if verbosity is None -> 0

    Args:
        * **verbosity** (:py:class:`int`): Verbosity level

    Returns:
        * **verbosity** (:py:class:`int`): Returns 0 if verbosity was initially `None`
    '''
    if verbosity is None:
        verbosity = 0
    return verbosity

# --------------------------
def check_noadaptind(noadaptind):
    '''
    Check if noadaptind is None -> Empty List

    Args:
        * **noadaptind** (:py:class:`list`): Indices not to be adapted in covariance matrix.

    Returns:
        * **noadaptind** (:py:class:`list`): Indices not to be adapted in covariance matrix.
    '''
    if noadaptind is None:
        noadaptind = []
    return noadaptind
# --------------------------
def noadapt_display_setting(ii, noadaptind):
    '''
    Define display settins if index not being adapted.

    Args:
        * **ii** (:py:class:`int`): Current index number
        * **noadaptind** (:py:class:`list`): List of indices not being adapted.

    Returns:
        * **st** (:py:class:`str`): String to be displayed, depending on if in `noadaptind`.
    '''
    if ii in noadaptind: # THIS PARAMETER IS FIXED
        st = ' (*)'
    else:
        st = ''
    return st
# --------------------------
def prior_display_setting(x):
    '''
    Define display string for prior.

    Args:
        * **x** (:py:class:`float`): Prior mean

    Returns:
        * **h2** (:py:class:`str`): String to be displayed, depending on if `x` is infinity.
    '''
    if math.isinf(x):
        h2 = ''
    else:
        h2 = '^2'
    return h2
# --------------------------    
def format_number_to_str(number):
    '''
    Format number for display

    Args:
        * **number** (:py:class:`float`): Number to be formatted

    Returns:
        * (:py:class:`str`): Formatted string display
    '''
    if abs(number) >= 1e4 or abs(number) <= 1e-2:
        return str('{:9.2e}'.format(number))
    else:
        return str('{:9.2f}'.format(number))
# --------------------------
def less_than_or_equal_to_zero(x):
    '''
    Return result of test on number based on less than or equal to

    Args:
        * **x** (:py:class:`float`): Number to be tested

    Returns:
        * (:py:class:`bool`): Result of test: `x<=0`
    '''
    return (x<=0)