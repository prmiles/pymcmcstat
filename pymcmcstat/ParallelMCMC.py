#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:58:22 2018

@author: prmiles
"""

from .MCMC import MCMC
from .ChainStatistics import ChainStatistics
from multiprocessing import Pool, cpu_count
import numpy as np
import sys
import os
import copy
import time

class ParallelMCMC:
    
    def __init__(self):
        self.description = 'Run MCMC simulations in parallel'
        
    def setup_parallel_simulation(self, mcset, initial_values = None, num_cores = 1, num_chain = 1):
    
        # extract settings from mcset
        data = mcset.data
        options = mcset.simulation_options
        model = mcset.model_settings
        parameters = mcset.parameters
        self.__get_parameter_features(parameters.parameters)
        options = self.__check_options_output(options)
        
        # number of CPUs
        self.__assign_number_of_cores(num_cores)
        
        # number of chains to generate
        self.num_chain = num_chain
        
        # check initial values
        self.__check_initial_values(initial_values)
        
        # assign parallel log file directory
        parallel_dir = options.savedir
        self.__check_directory(parallel_dir)
        self.__parallel_dir = parallel_dir
            
        # replicate features of mcstat object num_chain times
        self.parmc = []
        for ii in range(self.num_chain):
            
            self.parmc.append(MCMC())
            # replicate data
            self.parmc[ii].data = copy.deepcopy(data)
            # replicate model settings
            self.parmc[ii].model_settings = copy.deepcopy(model)
            # replicate simulation options and create log files for each
            self.parmc[ii].simulation_options = copy.deepcopy(options)
#            self.parmc[ii].simulation_options.save_to_txt = True
            chain_dir = str('chain_{}'.format(ii))
            self.parmc[ii].simulation_options.savedir = str('{}{}{}'.format(self.__parallel_dir,os.sep,chain_dir))
            # replicate parameter settings and assign distributed initial values
            self.parmc[ii].parameters = copy.deepcopy(parameters)
            for jj in range(self.npar):
                self.parmc[ii].parameters.parameters[jj]['theta0'] = self.initial_values[ii,jj]
            
    def run_parallel_simulation(self):
        start = time.time()
        mcpool = Pool(processes=self.num_cores) # depends on available cores
        res = mcpool.map(self._run_serial_simulation, self.parmc)
        mcpool.close() # not optimal! but easy
        mcpool.join()
        end = time.time()
        self.__parsimutime = end - start
        print('Parallel simulation run time: {} sec'.format(self.__parsimutime))
        
        # assign results to invidual simulations
        for ii in range(self.num_chain):
            self.parmc[ii].simulation_results = res[ii]
    
    def _run_serial_simulation(self, mcstat):
        print('Processing: {}'.format(mcstat.simulation_options.savedir))
        mcstat.run_simulation()
        return mcstat.simulation_results
        
    def display_individual_chain_statistics(self):
        CS = ChainStatistics()
        dividestr = '*********************'
        for ii in range(self.num_chain):
            print('\n{}\nDisplaying results for chain {}\nFiles: {}'.format(dividestr,ii,self.parmc[ii].simulation_options.savedir))
            res = self.parmc[ii].simulation_results.results
            chain = res['chain']
            CS.chainstats(chain,res)

    def __check_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    def __get_parameter_features(self, parameters):
        self.npar = len(parameters)
        self.__theta0 = np.zeros([self.npar])
        self.low_lim = np.zeros([self.npar])
        self.upp_lim = np.zeros([self.npar])
        for jj in range(self.npar):
            self.__theta0[jj] = parameters[jj]['theta0']
            self.low_lim[jj] = parameters[jj]['minimum']
            self.upp_lim[jj] = parameters[jj]['maximum']
            # check if infinity - needs to be finite for default mapping
            if self.low_lim[jj] == -np.inf:
                self.low_lim[jj] = self.__theta0[jj] - 100*(np.abs(self.__theta0[jj]))
                print('Finite lower limit required - setting low_lim[{}] = {}'.format(jj,self.low_lim[jj]))
            if self.upp_lim[jj] == np.inf:
                self.upp_lim[jj] = self.__theta0[jj] + 100*(np.abs(self.__theta0[jj]))
                print('Finite upper limit required - setting upp_lim[{}] = {}'.format(jj,self.upp_lim[jj]))
     
        
    def __check_options_output(self, options):
        if options.save_to_txt == False and options.save_to_bin == False:
            options.save_to_bin = True
        return options
        
    def __assign_number_of_cores(self, num_cores = 1):
        if num_cores > cpu_count():
            self.num_cores = cpu_count()
        else:
            self.num_cores = num_cores
            
    def __check_initial_values(self, initial_values):
        if initial_values is None:
            u = np.random.random_sample(size = (self.num_chain, self.npar))
            # map sampling between lower/upper limit
            self.initial_values = self.low_lim + (self.upp_lim - self.low_lim)*u
        else:
            m,n = initial_values.shape
            if m != self.num_chain:
                print('Shape of initial values inconsistent with requested number of chains.  \n num_chain = {}, initial_values.shape -> {},{}.  Resizing num_chain to {}.'.format(self.num_chain, m, n, m))
                self.num_chain = m
            if n != self.npar:
                print('Shape of initial values inconsistent with requested number of parameters.  \n npar = {}, initial_values.shape -> {},{}.  Only using first {} columns of initial_values.'.format(self.npar, m, n, self.npar))
                initial_values = np.delete(initial_values, [range(self.npar,n+1)], axis = 1)
            
            outsidebounds = self.__is_sample_outside_bounds(initial_values, self.low_lim, self.upp_lim)
            if outsidebounds is True:
                sys.exit(str('Initial values are not within parameter limits.  Make sure they are within the following limits:\n\tLower: {}\n\tUpper: {}\nThe initial_values tested were:\n{}'.format(self.low_lim, self.upp_lim, initial_values)))
            else:
                self.initial_values = initial_values
            
    def __is_sample_outside_bounds(self, theta, lower_limits, upper_limits):
        if (theta < lower_limits).any() or (theta > upper_limits).any():
            outsidebounds = True
        else:
            outsidebounds = False
        return outsidebounds
        
                