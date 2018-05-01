#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:18:19 2018

Description: Class used to organize results of MCMC simulation.

@author: prmiles
"""

# import required packages
import json
import numpy as np
from .NumpyEncoder import NumpyEncoder

class ResultsStructure:
    def __init__(self):
        self.results = {} # initialize empty dictionary
        self.basic = False # basic structure not add yet
     
    # --------------------------------------------------------
    def export_simulation_results_to_json_file(self, results, options):
                       
        if options.results_filename is None:
            dtstr = options.datestr
            filename = str('{}{}{}'.format(dtstr,'_','mcmc_simulation.json'))
        else:
            filename = options.results_filename
            
        self.save_json_object(results, filename)
    
    def save_json_object(self, results, filename):
        with open(filename, 'w') as out:
            json.dump(results, out, sort_keys=True, indent=4, cls=NumpyEncoder)
            
    def load_json_object(self, filename):
        with open(filename, 'r') as obj:
            results = json.load(obj)
        return results
    
    # --------------------------------------------------------
    def add_basic(self, options, model, covariance, parameters, rejected, simutime, theta):
        
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
            pass
    
    def add_prior(self, mu, sig, priorfun, priortype, priorpars):
        self.results['prior'] = [mu, sig]
        self.results['priorfun'] = priorfun
        self.results['priortype'] = priortype
        self.results['priorpars'] = priorpars
        
    def add_options(self, options = None):
        # Return options as dictionary
        opt = options.__dict__
        # define list of keywords to NOT add to results structure
        do_not_save_these_keys = ['doram', 'waitbar', 'debug', 'dodram', 'maxmem', 'verbosity', 'RDR', 'stats','initqcovn','drscale','maxiter','_SimulationOptions__options_set', 'skip']
        for ii in range(len(do_not_save_these_keys)):
            opt = self.removekey(opt, do_not_save_these_keys[ii])
            
        # must convert 'options' object to a dictionary
        self.results['simulation_options'] = opt

    def add_model(self, model = None):
        # Return model as dictionary
        mod = model.__dict__
        # define list of keywords to NOT add to results structure
        do_not_save_these_keys = ['sos_function','prior_function','model_function','prior_update_function','prior_pars']
        for ii in range(len(do_not_save_these_keys)):
            mod = self.removekey(mod, do_not_save_these_keys[ii])
        # must convert 'model' object to a dictionary
        self.results['model_settings'] = mod
        
    def add_chain(self, chain = None):
        self.results['chain'] = chain
        
    def add_s2chain(self, s2chain = None):
        self.results['s2chain'] = s2chain
        
    def add_sschain(self, sschain = None):
        self.results['sschain'] = sschain
        
    def add_time_stats(self, mtime, drtime, adtime):
        self.results['time [mh, dr, am]'] = [mtime, drtime, adtime]
        
    def add_random_number_sequence(self, rndseq):
        self.results['rndseq'] = rndseq
    
    def removekey(self, d, key):
        r = dict(d)
        del r[key]
        return r