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
import sys
#import os
import time
import numpy as np

from DataStructure import DataStructure
from ModelSettings import ModelSettings
from ModelParameters import ModelParameters
from SimulationOptions import SimulationOptions
from CovarianceProcedures import CovarianceProcedures
from ResultsStructure import ResultsStructure
from SumOfSquares import SumOfSquares
from PriorFunction import PriorFunction
from ParameterSet import ParameterSet
from SamplingMethods import SamplingMethods
from ErrorVarianceEstimator import ErrorVarianceEstimator
#import parameterfunctions as parfun
#import generalfunctions as genfun
#import mcmcfunctions as mcfun
#import selection_algorithms as selalg
from progressbar import progress_bar

class MCMC:
    def __init__(self):
        
        self.data = DataStructure()
        self.model_settings = ModelSettings()
        self.simulation_options = SimulationOptions()
        self.parameters = ModelParameters()
        self.covariance = CovarianceProcedures()
        self.sampling_methods = SamplingMethods()
        self.error_variance = ErrorVarianceEstimator()
            

    def run_simulation(self, previous_results = None):
        start_time = time.clock()
        
        self.__initialize_simulation()
        chain_index = 0
        self.__initialize_chains(chainind = chain_index)
        # ---------------------
        # setup progress bar
        if self.simulation_options.options.waitbar:
            wbarstatus = progress_bar(int(self.simulation_options.options.nsimu))
        # ---------------------
        # displacy current settings
        if self.simulation_options.options.verbosity >= 2:
            self.__display_current_mcmc_settings()        
        
        # ---------------------
        """
        Execute main simulator
        """
        self.__execute_simulator(chain_index = chain_index, wbarstatus = wbarstatus)
        
        end_time = time.clock()
        self.simulation_time = end_time - start_time

        # --------------------
        # Generate Results
        self.__generate_simulation_results()
      
        
    def __initialize_simulation(self):
        # ---------------------------------
        # check dependent parameters
        self.simulation_options.check_dependent_simulation_options(self.data, self.model_settings.model)
        self.model_settings.check_dependent_model_settings(self.data, self.simulation_options.options)
        
#        # ******************************************
#        # PENDING UPDATE
#        # use values from previous run (if inputted)
#        if previous_results != None:
#            genfun.message(self.options.verbosity, 0, '\nUsing values from the previous run')
#            self.parameters.results2params(previous_results, 1) # 1 = do local parameters
#            qcov = previous_results['qcov'] # previous_results should match the output format for the results class structure
#        # ******************************************
        
        # open and parse the parameter structure
        self.parameters.openparameterstructure(self.model_settings.model.nbatch)
        # check initial parameter values are inside range
        self.parameters.check_initial_values_wrt_parameter_limits()
        # add check that prior standard deviation > 0
        self.parameters.check_prior_sigma(self.simulation_options.options.verbosity)
        # display parameter settings
        self.parameters.display_parameter_settings(self.simulation_options.options)
        
        # setup covariance matrix and initial Cholesky decomposition
        self.covariance.initialize_covariance_settings(self.parameters, self.simulation_options.options)
        
        # ---------------------
        # define sum-of-squares object
        self.sos_object = SumOfSquares(self.model_settings.model, self.data, self.parameters)

        # ---------------------
        # define prior object
        self.prior_object = PriorFunction(priorfun = self.model_settings.model.prior_function, 
                                       mu = self.parameters.thetamu, 
                                       sigma = self.parameters.thetasigma)
        
        # ---------------------
        # Define initial parameter set
        self.initial_set = ParameterSet(theta = self.parameters.initial_value[self.parameters.parind[:]])
        
        # calculate sos with initial parameter set
        self.initial_set.ss = self.sos_object.evaluate_sos_function(self.initial_set.theta)
        nsos = len(self.initial_set.ss)
        
        # evaluate prior with initial parameter set
        self.initial_set.prior = self.prior_object.evaluate_prior(self.initial_set.theta)
        
        # add initial error variance to initial parameter set
        self.initial_set.sigma2 = self.model_settings.model.sigma2

        # recheck certain values in model settings that are dependent on the output of the sos function
        self.model_settings.check_dependent_model_settings_wrt_nsos(nsos)
        
        # ---------------------
        # Update variables covariance adaptation
        self.covariance.update_covariance_settings(self.initial_set.theta)
        
        if self.simulation_options.options.ntry > 1:
            self.sampling_methods.delayed_rejection.initialize_dr_metrics(self.simulation_options.options)    

    def __initialize_chains(self, chainind = 0):
        # ---------------------
        # Initialize chain, error variance, and SS
        self.chain = np.zeros([self.simulation_options.options.savesize, self.parameters.npar])
        self.sschain = np.zeros([self.simulation_options.options.savesize, self.model_settings.model.nsos])
        if self.simulation_options.options.updatesigma:
            self.s2chain = np.zeros([self.simulation_options.options.savesize, self.model_settings.model.nsos])
        else:
            self.s2chain = None
            
        # Save initialized values to chain, s2chain, sschain
#        chainind = 0 # where we are in chain
        self.chain[chainind,:] = self.initial_set.theta       
        self.sschain[chainind,:] = self.initial_set.ss
        if self.simulation_options.options.updatesigma:
            self.s2chain[chainind,:] = self.model_settings.model.sigma2
        
    def __execute_simulator(self, chain_index, wbarstatus):
        iiadapt = 0 # adaptation counter
        iiprint = 0 # print counter
        nsimu = self.simulation_options.options.nsimu
                
        self.rejected = {'total': 0, 'in_adaptation_interval': 0, 'outside_bounds': 0}
        self.old_set = self.initial_set
        
        for isimu in range(1, nsimu): # simulation loop
            # update indexing
            iiadapt += 1 # local adaptation index
            iiprint += 1 # local print index
            chain_index += 1
            # progress bar
            if self.simulation_options.options.waitbar:
                wbarstatus.update(isimu)
                
            self.__message(self.simulation_options.options.verbosity, 100, str('i:%d/%d\n'.format(isimu,nsimu)));
            
            # ---------------------------------------------------------
            # METROPOLIS ALGORITHM
            accept, new_set, outbound, npar_sample_from_normal = self.sampling_methods.metropolis.run_metropolis_step(
                    old_set = self.old_set, parameters = self.parameters, R = self.covariance.R, 
                    prior_object = self.prior_object, sos_object = self.sos_object)
    
            # --------------------------------------------------------
            # DELAYED REJECTION
            if self.simulation_options.options.ntry > 1 and accept == 0:
                # perform a new try according to delayed rejection 
                accept, new_set, outbound = self.sampling_methods.delayed_rejection.run_delayed_rejection(
                        old_set = self.old_set, new_set = new_set, RDR = self.covariance.RDR, ntry = self.simulation_options.options.ntry,
                        parameters = self.parameters, invR = self.covariance.invR, 
                        sosobj = self.sos_object, priorobj = self.prior_object)
    
            
            # --------------------------------------------------------
            # UPDATE CHAIN
            if accept:
                # accept
                self.chain[chain_index,:] = new_set.theta
                self.old_set = new_set
            else:
                # reject
                self.chain[chain_index,:] = self.old_set.theta
                self.rejected['total'] += 1
                self.rejected['in_adaptation_interval'] += 1
                if outbound:
                    self.rejected['outside_bounds'] += 1
                    
            # --------------------------------------------------------
            # PRINT REJECTION STATISTICS    
            if self.simulation_options.options.printint and iiprint + 1 == self.simulation_options.options.printint:
                self.__message(self.simulation_options.options.verbosity, 2, 
                               str('i:{} ({},{},{})\n'.format(isimu,
                                   self.rejected['total']*isimu**(-1)*100, self.rejected['in_adaptation_interval']*iiadapt**(-1)*100, 
                                   self.rejected['outside_bounds']*isimu**(-1)*100)))
                iiprint = 0 # reset print counter
                
            # UPDATE SUM-OF-SQUARES CHAIN
            self.sschain[chain_index,:] = self.old_set.ss
            
            # UPDATE ERROR VARIANCE
            if self.simulation_options.options.updatesigma:
                new_sigma2 = self.error_variance.update_error_variance(self.old_set.ss, self.model_settings.model)

                self.s2chain[chain_index,:] = new_sigma2
            
            # --------------------------------------------------------
            # ADAPTATION
            if self.simulation_options.options.adaptint > 0 and iiadapt == self.simulation_options.options.adaptint:
                self.covariance = self.sampling_methods.adaptation.run_adaptation(
                        covariance = self.covariance, options = self.simulation_options.options, 
                        isimu = isimu, iiadapt = iiadapt, rejected = self.rejected, 
                        chain = self.chain, chainind = chain_index, u = npar_sample_from_normal, 
                        npar = self.parameters.npar, new_set = new_set)
                
                iiadapt = 0 # reset local adaptation index
                self.rejected['in_adaptation_interval'] = 0 # reset local rejection index
       
    def __generate_simulation_results(self):
        # --------------------------------------------
        # BUILD RESULTS OBJECT
        self.simulation_results = ResultsStructure() # inititialize
            
        self.simulation_results.add_basic(options = self.simulation_options.options,
                                          model = self.model_settings.model,
                                          covariance = self.covariance,
                                          parameters = self.parameters,
                                          rejected = self.rejected, simutime = self.simulation_time, 
                                          theta = self.old_set.theta)
                
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
#        self.simulation_results.add_options(options = actual_options)
#        self.simulation_results.add_model(model = actual_model_settings)
        
        # add chain, s2chain, and sschain
        self.simulation_results.add_chain(chain = self.chain)
        self.simulation_results.add_s2chain(s2chain = self.s2chain)
        self.simulation_results.add_sschain(sschain = self.sschain)
        
        self.simulation_results.results # assign dictionary
        
    # display chain statistics
    def chainstats(self, chain, results = []):
        # 
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
