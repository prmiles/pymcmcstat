#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:18:19 2018

@author: prmiles
"""

# import required packages
import json
import numpy as np
from ..utilities.NumpyEncoder import NumpyEncoder
import os

class ResultsStructure:
    '''
    Results from MCMC simulation.
    
    **Description:** Class used to organize results of MCMC simulation.
    '''
    def __init__(self):
        self.results = {} # initialize empty dictionary
        self.basic = False # basic structure not add yet
     
    # --------------------------------------------------------
    def export_simulation_results_to_json_file(self, results):
        '''
        Export simulation results to a json file.
        
        :Args:
            * **results** (:class:`~.ResultsStructure`): Dictionary of MCMC simulation results/settings.
        '''
        savedir = results['simulation_options']['savedir']
        results_filename = results['simulation_options']['results_filename']
        if results_filename is None:
            dtstr = results['simulation_options']['datestr']
            filename = str('{}{}{}'.format(dtstr,'_','mcmc_simulation.json'))
        else:
            filename = results_filename
            
        self.save_json_object(results, os.path.join(savedir, filename))
    
    def save_json_object(self, results, filename):
        '''
        Save object to json file.
        
        .. note::
            
            Filename should include extension.
        
        :Args:
            * **results** (:py:class:`dict`): Object to save.
            * **filename** (:py:class:`str`): Write object into file with this name.
        '''
        with open(filename, 'w') as out:
            json.dump(results, out, sort_keys=True, indent=4, cls=NumpyEncoder)
            
    def load_json_object(self, filename):
        '''
        Load object stored in json file.
        
        .. note::
            
            Filename should include extension.
        
        :Args:
            * **filename** (:py:class:`str`): Load object from file with this name.
            
        \\
        
        :Returns:
            * **results** (:py:class:`dict`): Object loaded from file.
        '''
        with open(filename, 'r') as obj:
            results = json.load(obj)
        return results
    
    # --------------------------------------------------------
    def add_basic(self, options, model, covariance, parameters, rejected, simutime, theta):
        '''
        Add basic results from MCMC simulation to structure.
        
        :Args:
            * **options** (:class:`.SimulationOptions`): MCMC simulation options.
            * **model** (:class:`.ModelSettings`): MCMC model settings.
            * **covariance** (:class:`.CovarianceProcedures`): Covariance variables.
            * **parameters** (:class:`.ModelParameters`): Model parameters.
            * **rejected** (:py:class:`dict`): Dictionary of rejection stats.
            * **simutime** (:py:class:`float`): Simulation run time in seconds.
            * **theta** (:class:`~numpy.ndarray`): Last sampled values.
        '''
        
        self.results['theta'] = theta
        
        self.results['parind'] = parameters._parind
        self.results['local'] = parameters._local
        
        self.results['total_rejected'] = rejected['total']*(options.nsimu**(-1)) # total rejected
        self.results['rejected_outside_bounds'] = rejected['outside_bounds']*(options.nsimu**(-1)) # rejected due to sampling outside limits
        self.results['R'] = covariance._R
        self.results['qcov'] = np.dot(covariance._R.transpose(),covariance._R)
        self.results['cov'] = covariance._covchain
        self.results['mean'] = covariance._meanchain
        self.results['names'] = [parameters._names[ii] for ii in parameters._parind]
        self.results['limits'] = [parameters._lower_limits[parameters._parind[:]], parameters._upper_limits[parameters._parind[:]]]
             
        self.results['nsimu'] = options.nsimu
        self.results['simutime'] = simutime
        covariance._qcovorig[np.ix_(parameters._parind,parameters._parind)] = self.results['qcov']
        self.results['qcovorig'] = covariance._qcovorig
        self.basic = True # add_basic has been execute
        
    def add_updatesigma(self, updatesigma, sigma2, S20, N0):
        '''
        Add information to results structure related to observation error.
        
        :Args:
            * **updatesigma** (:py:class:`bool`): Flag to update error variance(s).
            * **sigma2** (:class:`~numpy.ndarray`): Latest estimate of error variance(s).
            * **S20** (:class:`~numpy.ndarray`): Scaling parameter(s).
            * **N0** (:class:`~numpy.ndarray`): Shape parameter(s).
            
        If :code:`updatesigma is True`, then
        
        ::
            
            results['sigma2'] = np.nan
            results['S20'] = S20
            results['N0'] = N0
            
        Otherwise
        
        ::
            
            results['sigma2'] = sigma2
            results['S20'] = np.nan
            results['N0'] = np.nan
        
        '''
        self.results['updatesigma'] = updatesigma
        if updatesigma:
            self.results['sigma2'] = np.nan
            self.results['S20'] = S20
            self.results['N0'] = N0
        else:
            self.results['sigma2'] = sigma2
            self.results['S20'] = np.nan
            self.results['N0'] = np.nan
    
    def add_dram(self, options, covariance, rejected, drsettings):
        '''
        Add results specific to performing DR algorithm.
        
        :Args:
            * **options** (:class:`.SimulationOptions`): MCMC simulation options.
            * **covariance** (:class:`.CovarianceProcedures`): Covariance variables.
            * **rejected** (:py:class:`dict`): Dictionary of rejection stats.
            * **drsettings** (:class:`~.DelayedRejection`): Need access to counters within DR class.
            
        '''
        # extract results from basic structure
        if self.basic is True:
            nsimu = self.results['nsimu']
            
            self.results['drscale'] = options.drscale
            
            rejected = rejected['total']
            drsettings.iacce[0] = nsimu - rejected - sum(drsettings.iacce[1:])
            # 1 - number accepted without DR, 2 - number accepted via DR try 1,
            # 3 - number accepted via DR try 2, etc.
            self.results['iacce'] = drsettings.iacce
            self.results['alpha_count'] = drsettings.dr_step_counter
            self.results['RDR'] = covariance._RDR
        else:
            print('Cannot add DRAM settings to results structure before running ''add_basic''')
    
    def add_prior(self, mu, sig, priorfun, priortype, priorpars):
        '''
        Add results specific to prior function.
        
        :Args:
            * **mu** (:py:class:`float`): Prior mean.
            * **sig** (:py:class:`float`): Prior standard deviation.
            * **priorfun**: Handle for prior function.
            * **priortype** (:py:class:`int`): Flag identifying type of prior.
            * **priorpars** (:py:class:`float`): Prior parameter for prior update function.
            
        .. note::
            
            This feature is not currently implemented.
        '''
        self.results['prior'] = [mu, sig]
        self.results['priorfun'] = priorfun
        self.results['priortype'] = priortype
        self.results['priorpars'] = priorpars
        
    def add_options(self, options = None):
        '''
        Saves subset of features of the simulation options in a nested dictionary.
        
        :Args:
            * **options** (:class:`.SimulationOptions`): MCMC simulation options.
        '''
        # Return options as dictionary
        opt = options.__dict__
        # define list of keywords to NOT add to results structure
        do_not_save_these_keys = ['doram', 'waitbar', 'debug', 'dodram', 'maxmem', 'verbosity', 'RDR', 'stats','initqcovn','drscale','maxiter','_SimulationOptions__options_set', 'skip']
        for ii in range(len(do_not_save_these_keys)):
            opt = self.removekey(opt, do_not_save_these_keys[ii])
            
        # must convert 'options' object to a dictionary
        self.results['simulation_options'] = opt

    def add_model(self, model = None):
        '''
        Saves subset of features of the model settings in a nested dictionary.
        
        :Args:
            * **model** (:class:`.ModelSettings`): MCMC model settings.
        '''
        # Return model as dictionary
        mod = model.__dict__
        # define list of keywords to NOT add to results structure
        do_not_save_these_keys = ['sos_function','prior_function','model_function','prior_update_function','prior_pars']
        for ii in range(len(do_not_save_these_keys)):
            mod = self.removekey(mod, do_not_save_these_keys[ii])
        # must convert 'model' object to a dictionary
        self.results['model_settings'] = mod
        
    def add_chain(self, chain = None):
        '''
        Add chain to results structure.
        
        :Args:
            * **chain** (:class:`~numpy.ndarray`): Model parameter sampling chain.
        '''
        self.results['chain'] = chain
        
    def add_s2chain(self, s2chain = None):
        '''
        Add observiation error chain to results structure.
        
        :Args:
            * **s2chain** (:class:`~numpy.ndarray`): Sampling chain of observation errors.
        '''
        self.results['s2chain'] = s2chain
        
    def add_sschain(self, sschain = None):
        '''
        Add sum-of-squares chain to results structure.
        
        :Args:
            * **sschain** (:class:`~numpy.ndarray`): Calculated sum-of-squares error for each parameter chains set.
        '''
        self.results['sschain'] = sschain
        
    def add_time_stats(self, mtime, drtime, adtime):
        '''
        Add time spend using each sampling algorithm.
        
        :Args:
            * **mtime** (:py:class:`float`): Time spent performing standard Metropolis.
            * **drtime** (:py:class:`float`): Time spent performing Delayed Rejection.
            * **adtime** (:py:class:`float`): Time spent performing Adaptation.
            
        .. note::
            
            This feature is not currently implemented.
        '''
        self.results['time [mh, dr, am]'] = [mtime, drtime, adtime]
        
    def add_random_number_sequence(self, rndseq):
        '''
        Add random number sequence to results structure.
        
        :Args:
            * **rndseq** (:class:`~numpy.ndarray`): Sequence of sampled random numbers.
            
        .. note::
            
            This feature is not currently implemented.
        '''
        self.results['rndseq'] = rndseq
    
    def removekey(self, d, key):
        '''
        Removed elements from dictionary and return the remainder.
        
        :Args:
            * **d** (:py:class:`dict`): Original dictionary.
            * **key** (:py:class:`str`): Keyword to be removed.
         
        \\
        
        :Returns:
            * **r** (:py:class:`dict`): Updated dictionary without the keywork, value pair.
        '''
        r = dict(d)
        del r[key]
        return r