#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17, 2018

Original files written in Matlab:
% Marko.Laine@helsinki.fi, 2003
% $Revision: 1.54 $  $Date: 2011/06/23 06:21:20 $
---------------
Python author: prmiles

Description: This module is intended to be the main class from which to run these
Markov Chain Monte Carlo type simulations.  The user will create an MCMC object,
initialize options, model settings, and parameters.  Simulations can then be run
as well as several types of predictive tests.
"""

# import required packages
#import sys
import os
import time
import numpy as np
import json
import datetime

from .DataStructure import DataStructure
from .ModelSettings import ModelSettings
from .ModelParameters import ModelParameters
from .SimulationOptions import SimulationOptions
from .CovarianceProcedures import CovarianceProcedures
from .ResultsStructure import ResultsStructure
from .SumOfSquares import SumOfSquares
from .PriorFunction import PriorFunction
from .ParameterSet import ParameterSet
from .SamplingMethods import SamplingMethods
from .ErrorVarianceEstimator import ErrorVarianceEstimator
from .MCMCPlotting import MCMCPlotting
from .NumpyEncoder import NumpyEncoder
from .PredictionIntervals import PredictionIntervals
from .progressbar import progress_bar

class MCMC:
    def __init__(self):
        
        # public variables
        self.data = DataStructure()
        self.model_settings = ModelSettings()
        self.simulation_options = SimulationOptions()
        self.parameters = ModelParameters()
        
        # private variables
        self._error_variance = ErrorVarianceEstimator()
        self._covariance = CovarianceProcedures()
        self._sampling_methods = SamplingMethods()
        
        self._mcmc_status = False
            

    def run_simulation(self, use_previous_results = False):
        start_time = time.clock()
        
        if use_previous_results == True:
            if self._mcmc_status == True:
                self.parameters._results_to_params(self.simulation_results.results, 1)
                self.__initialize_simulation()
                self.__expand_chains()
        else:
            self.__chain_index = 0 # start index at zero
            self.__initialize_simulation()
            self.__initialize_chains(chainind = self.__chain_index)
        # ---------------------
        # setup progress bar
        if self.simulation_options.waitbar:
            self.__wbarstatus = progress_bar(iters = int(self.simulation_options.nsimu))
            
        # ---------------------
        # displacy current settings
        if self.simulation_options.verbosity >= 2:
            self.__display_current_mcmc_settings()        
        
        # ---------------------
        """
        Execute main simulator
        """
        self.__execute_simulator()
        
        end_time = time.clock()
        self.__simulation_time = end_time - start_time

        # --------------------
        # Generate Results
        self.__generate_simulation_results()
        if self.simulation_options.save_to_json == True:
            self.__export_simulation_results_to_json_file(self.simulation_results.results)
        self.mcmcplot = MCMCPlotting()
        self.PI = PredictionIntervals()
        
        self._mcmc_status = True # simulation has been performed
        
    # display chain statistics
    def chainstats(self, chain = None, results = []):
        # 
        if chain is None:
            print('No chain reported - run simulation first.')
            pass
        else:
            m,n = chain.shape
            
            if results == []: # results is dictionary
                names = []
                for ii in range(n):
                    names.append(str('P{}'.format(ii)))
            else:
                names = results['names']
            
            meanii = []
            stdii = []
            for ii in range(n):
                meanii.append(np.mean(chain[:,ii]))
                stdii.append(np.std(chain[:,ii]))
                
            print('\n---------------------')
            print('{:10s}: {:>10s} {:>10s}'.format('name','mean','std'))
            for ii in range(n):
                if meanii[ii] > 1e4:
                    print('{:10s}: {:10.4g} {:10.4g}'.format(names[ii],meanii[ii],stdii[ii]))
                else:
                    print('{:10s}: {:10.4f} {:10.4f}'.format(names[ii],meanii[ii],stdii[ii]))
                    
    def __initialize_simulation(self):
        # ---------------------------------
        # check dependent parameters
        self.simulation_options._check_dependent_simulation_options(self.data, self.model_settings)
        self.model_settings._check_dependent_model_settings(self.data, self.simulation_options)
        
        # open and parse the parameter structure
        self.parameters._openparameterstructure(self.model_settings.nbatch)
        # check initial parameter values are inside range
        self.parameters._check_initial_values_wrt_parameter_limits()
        # add check that prior standard deviation > 0
        self.parameters._check_prior_sigma(self.simulation_options.verbosity)
        # display parameter settings
        self.parameters.display_parameter_settings(self.simulation_options)
        
        # setup covariance matrix and initial Cholesky decomposition
        self._covariance._initialize_covariance_settings(self.parameters, self.simulation_options)
        
        # ---------------------
        # define sum-of-squares object
        self.__sos_object = SumOfSquares(self.model_settings, self.data, self.parameters)

        # ---------------------
        # define prior object
        self.__prior_object = PriorFunction(priorfun = self.model_settings.prior_function, 
                                       mu = self.parameters._thetamu, 
                                       sigma = self.parameters._thetasigma)
        
        # ---------------------
        # Define initial parameter set
        self.__initial_set = ParameterSet(theta = self.parameters._initial_value[self.parameters._parind[:]])
        
        # calculate sos with initial parameter set
        self.__initial_set.ss = self.__sos_object.evaluate_sos_function(self.__initial_set.theta)
        nsos = len(self.__initial_set.ss)
        
        # evaluate prior with initial parameter set
        self.__initial_set.prior = self.__prior_object.evaluate_prior(self.__initial_set.theta)
        
        # add initial error variance to initial parameter set
        self.__initial_set.sigma2 = self.model_settings.sigma2

        # recheck certain values in model settings that are dependent on the output of the sos function
        self.model_settings._check_dependent_model_settings_wrt_nsos(nsos)
        
        # ---------------------
        # Update variables covariance adaptation
        self._covariance._update_covariance_settings(self.__initial_set.theta)
        
        if self.simulation_options.ntry > 1:
            self._sampling_methods.delayed_rejection._initialize_dr_metrics(self.simulation_options)    

    def __initialize_chains(self, chainind):
        # Initialize chain, error variance, and SS
        self.__chain = np.zeros([self.simulation_options.nsimu, self.parameters.npar])
        self.__sschain = np.zeros([self.simulation_options.nsimu, self.model_settings.nsos])
        if self.simulation_options.updatesigma:
            self.__s2chain = np.zeros([self.simulation_options.nsimu, self.model_settings.nsos])
        else:
            self.__s2chain = None
            
        # Save initialized values to chain, s2chain, sschain
        self.__chain[chainind,:] = self.__initial_set.theta       
        self.__sschain[chainind,:] = self.__initial_set.ss
        if self.simulation_options.updatesigma:
            self.__s2chain[chainind,:] = self.model_settings.sigma2
        
    def __expand_chains(self):
        # continuing simulation, so we must expand storage arrays
        zero_chain = np.zeros([self.simulation_options.nsimu-1, self.parameters.npar])
        zero_sschain = np.zeros([self.simulation_options.nsimu-1, self.model_settings.nsos])
        if self.simulation_options.updatesigma:
            zero_s2chain = np.zeros([self.simulation_options.nsimu-1, self.model_settings.nsos])
        else:
            zero_s2chain = None
            
        # Concatenate with previous chains
        self.__chain = np.concatenate((self.__chain, zero_chain), axis = 0)
        self.__sschain = np.concatenate((self.__sschain, zero_sschain), axis = 0)
        if self.simulation_options.updatesigma:
            self.__s2chain = np.concatenate((self.__s2chain, zero_s2chain), axis = 0)
        else:
            self.__s2chain = None
        
    def __execute_simulator(self):
        iiadapt = 0 # adaptation counter
        iiprint = 0 # print counter
        savecount = 0 # save counter
        lastbin = 0 # initialize bin counter
        nsimu = self.simulation_options.nsimu
                
        self.__rejected = {'total': 0, 'in_adaptation_interval': 0, 'outside_bounds': 0}
        self.__old_set = self.__initial_set
        
        for isimu in range(1, nsimu): # simulation loop
            # update indexing
            iiadapt += 1 # local adaptation index
            iiprint += 1 # local print index
            savecount += 1 # counter for saving to bin files
            self.__chain_index += 1
            # progress bar
            if self.simulation_options.waitbar:
                self.__wbarstatus.update(isimu)
                
            self.__message(self.simulation_options.verbosity, 100, str('i:%d/%d\n'.format(isimu,nsimu)));

            # METROPOLIS ALGORITHM
            accept, new_set, outbound, npar_sample_from_normal = self._sampling_methods.metropolis.run_metropolis_step(
                    old_set = self.__old_set, parameters = self.parameters, R = self._covariance._R, 
                    prior_object = self.__prior_object, sos_object = self.__sos_object)

            # DELAYED REJECTION
            if self.simulation_options.ntry > 1 and accept == 0:
                # perform a new try according to delayed rejection 
                accept, new_set, outbound = self._sampling_methods.delayed_rejection.run_delayed_rejection(
                        old_set = self.__old_set, new_set = new_set, RDR = self._covariance._RDR, ntry = self.simulation_options.ntry,
                        parameters = self.parameters, invR = self._covariance._invR, 
                        sosobj = self.__sos_object, priorobj = self.__prior_object)

            # UPDATE CHAIN
            self.__update_chain(accept = accept, new_set = new_set, outbound = outbound)

            # PRINT REJECTION STATISTICS    
            if self.simulation_options.printint and iiprint + 1 == self.simulation_options.printint:
                self.__print_rejection_statistics(isimu = isimu, iiadapt = iiadapt, verbosity = self.simulation_options.verbosity)
                iiprint = 0 # reset print counter
                
            # UPDATE SUM-OF-SQUARES CHAIN
            self.__sschain[self.__chain_index,:] = self.__old_set.ss
            
            # UPDATE ERROR VARIANCE
            if self.simulation_options.updatesigma:
                sigma2 = self._error_variance.update_error_variance(self.__old_set.ss, self.model_settings)
                self.__s2chain[self.__chain_index,:] = sigma2
                self.__old_set.sigma2 = sigma2

            # ADAPTATION
            if self.simulation_options.adaptint > 0 and iiadapt == self.simulation_options.adaptint:
                self._covariance = self._sampling_methods.adaptation.run_adaptation(
                        covariance = self._covariance, options = self.simulation_options, 
                        isimu = isimu, iiadapt = iiadapt, rejected = self.__rejected, 
                        chain = self.__chain, chainind = self.__chain_index, u = npar_sample_from_normal, 
                        npar = self.parameters.npar, new_set = new_set)
                
                iiadapt = 0 # reset local adaptation index
                self.__rejected['in_adaptation_interval'] = 0 # reset local rejection index
                
            # SAVE TO BIN FILE
            if self.simulation_options.save_to_bin is True and savecount == self.simulation_options.savesize:
                savesize = self.simulation_options.savesize
                start = isimu - savesize
                end = isimu
                self.__save_chains_to_bin(start, end)
                # reset counter
                savecount = 0
                lastbin = isimu
       
        # SAVE REMAINING ELEMENTS TO BIN FILE
        if self.simulation_options.save_to_bin is True:
            start = lastbin
            end = isimu + 1
            self.__save_chains_to_bin(start, end)
           
        # update value to end value
        self.parameters._value[self.parameters._parind] = self.__old_set.theta
           
    def __generate_simulation_results(self):
        # --------------------------------------------
        # BUILD RESULTS OBJECT
        self.simulation_results = ResultsStructure() # inititialize
            
        self.simulation_results.add_basic(options = self.simulation_options,
                                          model = self.model_settings,
                                          covariance = self._covariance,
                                          parameters = self.parameters,
                                          rejected = self.__rejected, simutime = self.__simulation_time, 
                                          theta = self.parameters._value)
                
#        self.simulation_results.add_updatesigma(updatesigma = updatesigma, sigma2 = sigma2,
#                                S20 = S20, N0 = N0)
#        
#        self.simulation_results.add_prior(mu = thetamu[parind[:]], sig = thetasig[parind[:]], 
#                          priorfun = priorfun, priortype = priortype, 
#                          priorpars = priorpars)
#        
#        if dodram == 1:
#            self.simulation_results.add_dram(dodram = dodram, drscale = drscale, iacce = iacce,
#                         alpha_count = A_count, RDR = RDR, nsimu = nsimu, rej = rej)
#        
        self.simulation_results.add_options(options = self.simulation_options)
        self.simulation_results.add_model(model = self.model_settings)
        
        # add chain, s2chain, and sschain
        self.simulation_results.add_chain(chain = self.__chain)
        self.simulation_results.add_s2chain(s2chain = self.__s2chain)
        self.simulation_results.add_sschain(sschain = self.__sschain)
        
        self.simulation_results.results # assign dictionary
    
    def __save_chains_to_bin(self, start, end):
#        print('start = {}, end = {}'.format(start, end))
        
        savedir = self.simulation_options.savedir
        
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        
        chainfile = os.path.join(savedir, self.simulation_options.chainfile)
        s2chainfile = os.path.join(savedir, self.simulation_options.s2chainfile)
        sschainfile = os.path.join(savedir, self.simulation_options.sschainfile)
        covchainfile = os.path.join(savedir, self.simulation_options.covchainfile)
        
        binlogfile = os.path.join(savedir, 'binlogfile.txt')
        binstr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.__add_to_bin_log(binlogfile, str('{}\t{}\t{}\n'.format(binstr, start, end-1)))
        
        self.__save_to_bin_file(chainfile, self.__chain[start:end,:]) 
        self.__save_to_bin_file(sschainfile, self.__sschain[start:end,:]) 
        self.__save_to_bin_file(covchainfile, np.dot(self._covariance._R.transpose(),self._covariance._R), True)
        if self.simulation_options.updatesigma == 1:
            self.__save_to_bin_file(s2chainfile, self.__s2chain[start:end,:]) 

    def __add_to_bin_log(self, filename, binstr):
        with open(filename, 'a') as binfile:
            binfile.write(binstr)
    
    def __save_to_bin_file(self, filename, mtx, separate_set = False):
        if separate_set:
            with open(filename, "a") as text_file:
                text_file.write('--------------------------\n')
                
        handle = open(filename, 'a')
        np.savetxt(handle,mtx)
        handle.close()
    
    def __export_simulation_results_to_json_file(self, results = None):
                       
        if self.simulation_options.results_filename is None:
            dtstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = str('{}{}{}'.format(dtstr,'_','mcmc_simulation.json'))
        else:
            filename = self.simulation_options.results_filename
            
        #save_dill_object(mcstat, filename)
        self.__save_json_object(results, filename)
    
    def __save_json_object(self, obj, filename):
        with open(filename, 'w') as out:
            json.dump(obj, out, sort_keys=True, indent=4, cls=NumpyEncoder)
            
    def load_json_object(self, filename):
        with open(filename, 'r') as obj:
            results = json.load(obj)
        return results
    
    def __update_chain(self, accept, new_set, outbound):
        if accept:
            # accept
            self.__chain[self.__chain_index,:] = new_set.theta
            self.__old_set = new_set
        else:
            # reject
            self.__chain[self.__chain_index,:] = self.__old_set.theta
            self.__update_rejected(outbound)
            
    def __print_rejection_statistics(self, isimu, iiadapt, verbosity):
        self.__message(verbosity, 2, str('i:{} ({},{},{})\n'.format(
                isimu, self.__rejected['total']*isimu**(-1)*100, self.__rejected['in_adaptation_interval']*iiadapt**(-1)*100, 
                self.__rejected['outside_bounds']*isimu**(-1)*100)))
                        
    def __message(self, verbosity, level, printthis):
        printed = False
        if verbosity >= level:
            print(printthis)
            printed = True
        return printed
    
    def __display_current_mcmc_settings(self):
        self.model_settings.display_model_settings()
        self.simulation_options.display_simulation_options()
        self.covariance.display_covariance_settings()

    def __update_rejected(self, outbound):
        self.__rejected['total'] += 1
        self.__rejected['in_adaptation_interval'] += 1
        if outbound:
            self.__rejected['outside_bounds'] += 1