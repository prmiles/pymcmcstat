#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:58:22 2018

@author: prmiles
"""

from .MCMC import MCMC
from .chain import ChainStatistics as CS
from .samplers.utilities import is_sample_outside_bounds
from multiprocessing import Pool, cpu_count
import numpy as np
import sys
import os
import copy
import time

class ParallelMCMC:
    '''
    Run Parallel MCMC Simulations.

    Attributes:
        * :meth:`~setup_parallel_simulation`
        * :meth:`~run_parallel_simulation`
        * :meth:`~display_individual_chain_statistics`
    '''
    def __init__(self):
        self.description = 'Run MCMC simulations in parallel'

    def setup_parallel_simulation(self, mcset, initial_values = None, num_cores = 1, num_chain = 1):
        '''
        Setup simulation to run in parallel.
        
        Settings defined in `mcset` object will be copied into different instances in
        order to run parallel chains.
        
        Args:
            * **mcset** (:class:`~.MCMC`): Instance of MCMC object with serial simulation setup.
            * **initial_values** (:class:`~numpy.ndarray`): Array of initial parameter values - [num_chain,npar].
            * **num_cores** (:py:class:`int`): Number of cores designated by user.
            * **num_chain** (:py:class:`int`): Number of sampling chains to be generated.
        '''
        # extract settings from mcset
        data, options, model, parameters = unpack_mcmc_set(mcset = mcset)
        npar, low_lim, upp_lim = get_parameter_features(parameters.parameters)
        options = check_options_output(options)
        
        # number of CPUs
        self.num_cores = assign_number_of_cores(num_cores)
        
        # check initial values and number of chains to be generated
        self.initial_values, self.num_chain = check_initial_values(initial_values = initial_values, num_chain = num_chain, npar = npar, low_lim = low_lim, upp_lim = upp_lim)
        
        # assign parallel log file directory
        parallel_dir = options.savedir
        check_directory(parallel_dir)
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
            chain_dir = str('chain_{}'.format(ii))
            self.parmc[ii].simulation_options.savedir = str('{}{}{}'.format(self.__parallel_dir,os.sep,chain_dir))
            # replicate parameter settings and assign distributed initial values
            self.parmc[ii].parameters = copy.deepcopy(parameters)
            for jj in range(npar):
                self.parmc[ii].parameters.parameters[jj]['theta0'] = self.initial_values[ii,jj]
            
    def run_parallel_simulation(self):
        '''
        Run MCMC simulations in parallel.
        
        The code is run in parallel by using :class:`~.Pool`.  While
        running, you can expect a display similar to

        ::
            
            Processing: <parallel_dir>/chain_1
            Processing: <parallel_dir>/chain_0
            Processing: <parallel_dir>/chain_2
            
        The simulation is complete when you see the run time displayed.
        
        ::
            
            Parallel simulation run time: 16.15234899520874 sec
        '''
        start = time.time()
        mcpool = Pool(processes=self.num_cores) # depends on available cores
        res = mcpool.map(run_serial_simulation, self.parmc)
        mcpool.close() # not optimal! but easy
        mcpool.join()
        end = time.time()
        self.__parsimutime = end - start
        print('Parallel simulation run time: {} sec'.format(self.__parsimutime))
        # assign results to invidual simulations
        for ii in range(self.num_chain):
            self.parmc[ii].simulation_results = res[ii]

    # -------------------------
    def display_individual_chain_statistics(self):
        '''
        Display chain statistics for different chains in parallel simulation.
        
        Example display:

        ::
            
            ****************************************
            Displaying results for chain 0
            Files: <parallel_dir>/chain_0
            
            ---------------------
            name      :       mean        std     MC_err        tau     geweke
            m         :     1.9869     0.1295     0.0107   320.0997     0.9259
            b         :     3.0076     0.2489     0.0132   138.1260     0.9413
            ---------------------
            
            ****************************************
            Displaying results for chain 1
            Files: <parallel_dir>/chain_1
            
            ---------------------
            name      :       mean        std     MC_err        tau     geweke
            m         :     1.8945     0.4324     0.0982  2002.6361     0.3116
            b         :     3.2240     1.0484     0.2166  1734.0201     0.4161
            ---------------------
        '''
        for ii, mc in enumerate(self.parmc):
            print('\n{}\nDisplaying results for chain {}\nFiles: {}'.format(40*'*',ii,mc.simulation_options.savedir))
            CS.chainstats(mc.simulation_results.results['chain'],mc.simulation_results.results)

# -------------------------
def unpack_mcmc_set(mcset):
    '''
    Unpack attributes of MCMC object.
    
    Args:
        * **mcset** (:class:`~.MCMC`): MCMC object.
        
    Returns:
        * **data** (:class:`~.DataStructure`): MCMC data structure.
        * **options** (:class:`~.SimulationOptions`): MCMC simulation options.
        * **model** (:class:`~.ModelSettings`): MCMC model settings.
        * **parameters** (:class:`~.ModelParameters`): MCMC model parameters.
    '''
    data = mcset.data
    options = mcset.simulation_options
    model = mcset.model_settings
    parameters = mcset.parameters
    return data, options, model, parameters
# -------------------------
def get_parameter_features(parameters):
    '''
    Get features of model parameters.
    
    Args:
        * **parameters** (:py:class:`list`): List of MCMC model parameter dictionaries.

    Returns:
        * **npar** (:py:class:`int`): Number of model parameters.
        * **low_lim** (:class:`~numpy.ndarray`): Lower limits.
        * **upp_lim** (:class:`~numpy.ndarray`): Upper limits.
    '''
    npar = len(parameters)
    theta0 = np.zeros([npar])
    low_lim = np.zeros([npar])
    upp_lim = np.zeros([npar])
    for jj in range(npar):
        theta0[jj] = parameters[jj]['theta0']
        low_lim[jj] = parameters[jj]['minimum']
        upp_lim[jj] = parameters[jj]['maximum']
        # check if infinity - needs to be finite for default mapping
        if low_lim[jj] == -np.inf:
            low_lim[jj] = theta0[jj] - 100*(np.abs(theta0[jj]))
            print('Finite lower limit required - setting low_lim[{}] = {}'.format(jj,low_lim[jj]))
        if upp_lim[jj] == np.inf:
            upp_lim[jj] = theta0[jj] + 100*(np.abs(theta0[jj]))
            print('Finite upper limit required - setting upp_lim[{}] = {}'.format(jj,upp_lim[jj]))
    return npar, low_lim, upp_lim
# -------------------------
def check_initial_values(initial_values, num_chain, npar, low_lim, upp_lim):
    '''
    Check if initial values satisfy requirements.
    
    Args:
        * **initial_values** (:class:`~numpy.ndarray`): Array of initial parameter values - [num_chain,npar].
        * **num_chain** (:py:class:`int`): Number of sampling chains to be generated.
        * **npar** (:py:class:`int`): Number of model parameters.
        * **low_lim** (:class:`~numpy.ndarray`): Lower limits.
        * **upp_lim** (:class:`~numpy.ndarray`): Upper limits.

    Returns:
        * **initial_values** (:class:`~numpy.ndarray`): Array of initial parameter values - [num_chain,npar].
        * **num_chain** (:py:class:`int`): Number of sampling chains to be generated.
    '''
    if initial_values is None:
        initial_values = generate_initial_values(num_chain = num_chain, npar = npar, low_lim = low_lim, upp_lim = upp_lim)
    else:
        num_chain, initial_values = check_shape_of_users_initial_values(initial_values = initial_values, num_chain = num_chain, npar = npar)
        initial_values = check_users_initial_values_wrt_limits(initial_values = initial_values, low_lim = low_lim, upp_lim = upp_lim)
    return initial_values, num_chain
# -------------------------
def generate_initial_values(num_chain, npar, low_lim, upp_lim):
    '''
    Generate initial values by sampling from uniform distribution between limits
    
    Args:
        * **num_chain** (:py:class:`int`): Number of sampling chains to be generated.
        * **npar** (:py:class:`int`): Number of model parameters.
        * **low_lim** (:class:`~numpy.ndarray`): Lower limits.
        * **upp_lim** (:class:`~numpy.ndarray`): Upper limits.

    Returns:
        * **initial_values** (:class:`~numpy.ndarray`): Array of initial parameter values - [num_chain,npar]
    '''
    u = np.random.random_sample(size = (num_chain, npar))
    # map sampling between lower/upper limit
    initial_values = low_lim + (upp_lim - low_lim)*u
    return initial_values
# -------------------------
def check_shape_of_users_initial_values(initial_values, num_chain, npar):
    '''
    Check shape of users initial values
    
    Args:
        * **initial_values** (:class:`~numpy.ndarray`): Array of initial parameter values - expect [num_chain,npar]
        * **num_chain** (:py:class:`int`): Number of sampling chains to be generated.
        * **npar** (:py:class:`int`): Number of model parameters.

    Returns:
        * **num_chain** (:py:class:`int`): Number of sampling chains to be generated - equal to number of rows in initial values array.
        * **initial_values**
    '''
    m,n = initial_values.shape
    if m != num_chain:
        print('Shape of initial values inconsistent with requested number of chains.  \n num_chain = {}, initial_values.shape -> {},{}.  Resizing num_chain to {}.'.format(num_chain, m, n, m))
        num_chain = m
    if n != npar:
        print('Shape of initial values inconsistent with requested number of parameters.  \n npar = {}, initial_values.shape -> {},{}.  Only using first {} columns of initial_values.'.format(npar, m, n, npar))
        initial_values = np.delete(initial_values, [range(npar,n+1)], axis = 1)
    return num_chain, initial_values
# -------------------------
def check_users_initial_values_wrt_limits(initial_values, low_lim, upp_lim):
    '''
    Check users initial values wrt parameter limits
    
    Args:
        * **initial_values** (:class:`~numpy.ndarray`): Array of initial parameter values - expect [num_chain,npar]
        * **low_lim** (:class:`~numpy.ndarray`): Lower limits.
        * **upp_lim** (:class:`~numpy.ndarray`): Upper limits.

    Returns:
        * **initial_values** (:class:`~numpy.ndarray`): Array of initial parameter values - expect [num_chain,npar]

    Raises:
        * `SystemExit` if initial values are outside parameter bounds.
    '''
    outsidebounds = is_sample_outside_bounds(initial_values, low_lim, upp_lim)
    if outsidebounds is True:
        sys.exit(str('Initial values are not within parameter limits.  Make sure they are within the following limits:\n\tLower: {}\n\tUpper: {}\nThe initial_values tested were:\n{}'.format(low_lim, upp_lim, initial_values)))
    else:
        return initial_values
# -------------------------
def check_options_output(options):
    '''
    Check output settings defined in options
    
    Args:
        * **options** (:class:`.SimulationOptions`): MCMC simulation options.

    Returns:
        * **options** (:class:`.SimulationOptions`): MCMC simulation options with at least binary save flag set to True.
    '''
    if options.save_to_txt == False and options.save_to_bin == False:
        options.save_to_bin = True
    return options

# -------------------------
def check_directory(directory):
    '''
    Check and make sure directory exists
    
    Args:
        * **directory** (:py:class:`str`): Folder/directory path name.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
        
# -------------------------
def run_serial_simulation(mcstat):
    '''
    Run serial MCMC simulation
    
    Args:
        * **mcstat** (:class:`MCMC.MCMC`): MCMC object.

    Returns:
        * **results** (:py:class:`dict`): Results dictionary for serial simulation.
    '''
    print('Processing: {}'.format(mcstat.simulation_options.savedir))
    mcstat.run_simulation()
    return mcstat.simulation_results

# -------------------------
def assign_number_of_cores(num_cores = 1):
    '''
    Assign number of cores to use in parallel process
    
    Args:
        * **num_cores** (:py:class:`int`): Number of cores designated by user.

    Returns:
        * **num_cores** (:py:class:`int`): Number of cores designated by user or maximum number of cores available on machine.
    '''
    tmp = cpu_count()
    if num_cores > tmp:
        return tmp
    else:
        return num_cores