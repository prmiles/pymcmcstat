#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17, 2018

@author: prmiles

Description: This module is intended to be the main class from which to run these
Markov Chain Monte Carlo type simulations.  The user will create an MCMC object,
initialize data, simulation options, model settings and parameters.
"""

# import required packages
import os
import time
import numpy as np
import datetime
import sys

from .settings.DataStructure import DataStructure
from .settings.ModelSettings import ModelSettings
from .settings.ModelParameters import ModelParameters
from .settings.SimulationOptions import SimulationOptions

from .procedures.CovarianceProcedures import CovarianceProcedures
from .procedures.SumOfSquares import SumOfSquares
from .procedures.PriorFunction import PriorFunction
from .procedures.ErrorVarianceEstimator import ErrorVarianceEstimator

from .structures.ParameterSet import ParameterSet
from .structures.ResultsStructure import ResultsStructure

from .samplers.SamplingMethods import SamplingMethods

from .plotting import MCMCPlotting
from .plotting.PredictionIntervals import PredictionIntervals

from .chain import ChainStatistics
from .chain import ChainProcessing

from .utilities.progressbar import progress_bar
from .utilities.general import message


# --------------------------------------------------------
class MCMC:
    '''
    Markov Chain Monte Carlo (MCMC) simulation object.

    This class type is the primary feature of this Python package.  All simulations
    are run through this class type, and for the most part the user will interact
    with an object of this type.  The class initialization provides the option for
    setting the random seed, which can be very useful for testing the functionality
    of the code.  It was found that setting the random seed at object initialization
    was the simplest interface.

    Args:
        * **rngseed** (:py:class:`float`): Seed for numpy's random number generator.

    Attributes:
        * :meth:`~run_simulation`
        * :meth:`~display_current_mcmc_settings`
        * **data** (:class:`~.DataStructure`): MCMC data structure.
        * **simulation_options** (:class:`~.SimulationOptions`): MCMC simulation options.
        * **model_settings** (:class:`~.ModelSettings`): MCMC model settings.
        * **parameters** (:class:`~.ModelParameters`): MCMC model parameters.
    '''
    def __init__(self, rngseed=None, seterr={'over': 'ignore', 'under': 'ignore'}):
        # public variables
        self.data = DataStructure()
        self.model_settings = ModelSettings()
        self.simulation_options = SimulationOptions()
        self.parameters = ModelParameters()
        self.custom_samplers = []
        # private variables
        self._error_variance = ErrorVarianceEstimator()
        self._covariance = CovarianceProcedures()
        self._sampling_methods = SamplingMethods()
        self._mcmc_status = False
        np.random.seed(seed=rngseed)
        np.seterr(**seterr)

    # --------------------------------------------------------
    def run_simulation(self, use_previous_results=False):
        '''
        Run MCMC Simulation

        .. note::

            This is the method called by the user to run the simulation.  The user
            must specify a data structure, setup simulation options, and define
            the model settings and parameters before calling this method.

        Args:
            * **use_previous_results** (:py:class:`bool`): Flag to indicate whether simulation is being restarted.
        '''
        start_time = time.time()
        self.__setup_simulator(use_previous_results=use_previous_results)
        # ---------------------
        # setup progress bar
        if self.simulation_options.waitbar:
            self.__wbarstatus = progress_bar(iters=int(self.simulation_options.nsimu))
        # ---------------------
        # displacy current settings
        if self.simulation_options.verbosity >= 2:
            self.display_current_mcmc_settings()
        # ---------------------
        # Execute main simulator
        self.__execute_simulator()
        end_time = time.time()
        self.__simulation_time = end_time - start_time
        # --------------------
        # Generate Results
        self.__generate_simulation_results()
        if self.simulation_options.save_to_json is True:
            if self.simulation_results.basic is True:  # check that results structure has been created
                self.simulation_results.export_simulation_results_to_json_file(results=self.simulation_results.results)
        self.mcmcplot = MCMCPlotting.Plot()
        self.PI = PredictionIntervals()
        self.chainstats = ChainStatistics.chainstats
        self._mcmc_status = True  # simulation has been performed

    # --------------------------------------------------------
    def __setup_simulator(self, use_previous_results):
        '''
        Setup simulator.

        If previous results exist, then chain arrays will be expanded.  Otherwise,
        they will be initialized.

        If `use_previous_results` is `False`, one can still restart a simulation
        by specifying a `json_restart_file` in the :class:`~.SimulationOptions`.

        Args:
            * **use_previous_results** (:py:class:`bool`): Flag to indicate whether simulation is being restarted.
        '''
        if use_previous_results is True:
            if self._mcmc_status is True:
                self.parameters._results_to_params(self.simulation_results.results, 1)
                self._initialize_simulation()
                self.__expand_chains(
                        nsimu=self.simulation_options.nsimu,
                        npar=self.parameters.npar,
                        nsos=self.model_settings.nsos,
                        updatesigma=self.simulation_options.updatesigma)
            else:
                sys.exit('No previous results found.  Set ''use_previous_results'' to ''False''')
        else:
            if self.simulation_options.json_restart_file is not None:
                RS = ResultsStructure()
                res = RS.load_json_object(self.simulation_options.json_restart_file)
                self.parameters._results_to_params(res, 1)
                self.simulation_options.qcov = np.array(res['qcov'])

            self.__chain_index = 0  # start index at zero
            self._initialize_simulation()
            self.__initialize_chains(
                    chainind=self.__chain_index,
                    nsimu=self.simulation_options.nsimu,
                    npar=self.parameters.npar,
                    nsos=self.model_settings.nsos,
                    updatesigma=self.simulation_options.updatesigma,
                    sigma2=self.model_settings.sigma2)

    # --------------------------------------------------------
    def _initialize_simulation(self):
        '''
        Initialize all dependent settings for simulation.
        '''
        # ---------------------------------
        # check dependent parameters
        self.simulation_options._check_dependent_simulation_options(self.model_settings)
        self.model_settings._check_dependent_model_settings(self.data, self.simulation_options)
        # open and parse the parameter structure
        self.parameters._openparameterstructure(self.model_settings.nbatch)
        # check initial parameter values are inside range
        self.parameters._check_initial_values_wrt_parameter_limits()
        # add check that prior standard deviation > 0
        self.parameters._check_prior_sigma(self.simulation_options.verbosity)
        # display parameter settings
        self.parameters.display_parameter_settings(self.simulation_options.verbosity, self.parameters._no_adapt)
        # setup covariance matrix and initial Cholesky decomposition
        self._covariance._initialize_covariance_settings(self.parameters, self.simulation_options)
        # ---------------------
        # define sum-of-squares object
        self.__sos_object = SumOfSquares(self.model_settings, self.data, self.parameters)
        # ---------------------
        # define prior object
        self.__prior_object = PriorFunction(
                priorfun=self.model_settings.prior_function,
                mu=self.parameters._thetamu[self.parameters._parind[:]],
                sigma=self.parameters._thetasigma[self.parameters._parind[:]])
        # ---------------------
        # Define initial parameter set
        self.__initial_set = ParameterSet(theta=self.parameters._initial_value[self.parameters._parind[:]])
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
        # ---------------------
        # Setup custom samplers
        self.custom_sampler_output = []
        if len(self.custom_samplers) > 0:
            for ii, cs in enumerate(self.custom_samplers):
                self.custom_sampler_output.append(cs.setup())

    # --------------------------------------------------------
    def __initialize_chains(self, chainind, nsimu, npar, nsos, updatesigma, sigma2):
        '''
        Initialize chains

        Args:
            * **chainind** (:py:class:`int`): Where to store initial parameter value
            * **nsimu** (:py:class:`int`): Number of parameter samples to simulate.  Default is 1e4.
            * **npar** (:py:class:`int`): Number of parameters being sampled.
            * **nsos** (:py:class:`int`): Length of output from sum-of-squares function
            * **updatesigma** (:py:class:`bool`): Flag for updating measurement error variance. \
            Default is 0 -> off (1 -> on).
            * **sigma2** (:class:`numpy.ndarray`): Initial error observations.
        '''
        # Initialize chain, error variance, and SS
        self.__chain = np.zeros([nsimu, npar])
        self.__sschain = np.zeros([nsimu, nsos])
        # Save initialized values to chain, s2chain, sschain
        self.__chain[chainind, :] = self.__initial_set.theta
        self.__sschain[chainind, :] = self.__initial_set.ss

        self.__chains = []
        self.__chains.append(dict(file=self.simulation_options.chainfile, mtx=self.__chain))
        self.__chains.append(dict(file=self.simulation_options.sschainfile, mtx=self.__sschain))

        if updatesigma:
            self.__s2chain = np.zeros([nsimu, nsos])
            self.__s2chain[chainind, :] = sigma2
            self.__chains.append(dict(file=self.simulation_options.s2chainfile, mtx=self.__s2chain))
        else:
            self.__s2chain = None

    # --------------------------------------------------------
    def __expand_chains(self, nsimu, npar, nsos, updatesigma):
        '''
        Expand chains for extended simulation

        Args:
            * **nsimu** (:py:class:`int`): Number of parameter samples to simulate.  Default is 1e4.
            * **npar** (:py:class:`int`): Number of parameters being sampled.
            * **nsos** (:py:class:`int`): Length of output from sum-of-squares function
            * **updatesigma** (:py:class:`bool`): Flag for updating measurement error variance. \
            Default is 0 -> off (1 -> on).
        '''
        # continuing simulation, so we must expand storage arrays
        zero_chain = np.zeros([nsimu-1, npar])
        zero_sschain = np.zeros([nsimu-1, nsos])
        # Concatenate with previous chains
        self.__chain = np.concatenate((self.__chain, zero_chain), axis=0)
        self.__sschain = np.concatenate((self.__sschain, zero_sschain), axis=0)
        if updatesigma:
            zero_s2chain = np.zeros([nsimu-1, nsos])
            self.__s2chain = np.concatenate((self.__s2chain, zero_s2chain), axis=0)
        else:
            self.__s2chain = None

    # --------------------------------------------------------
    def __execute_simulator(self):
        '''
        Execute MCMC simulation.

        Runs through `nsimu` simulations using `method` defined by user.
        '''
        iiadapt = 0  # adaptation counter
        iiprint = 0  # print counter
        savecount = 0  # save counter
        lastbin = 0  # initialize bin counter
        nsimu = self.simulation_options.nsimu

        self.__rejected = {'total': 0, 'in_adaptation_interval': 0, 'outside_bounds': 0}
        self.__old_set = self.__initial_set

        for isimu in range(1, nsimu):  # simulation loop
            # update indexing
            iiadapt += 1  # local adaptation index
            iiprint += 1  # local print index
            savecount += 1  # counter for saving to bin files
            self.__chain_index += 1
            # progress bar
            if self.simulation_options.waitbar:
                self.__wbarstatus.update(isimu)

            message(self.simulation_options.verbosity, 100, str('i: {:d}/{:d}\n'.format(isimu, nsimu)))

            # METROPOLIS
            accept, new_set, outbound, npar_sample_from_normal = self._sampling_methods.metropolis.run_metropolis_step(
                    old_set=self.__old_set,
                    parameters=self.parameters,
                    R=self._covariance._R,
                    prior_object=self.__prior_object,
                    sos_object=self.__sos_object,
                    custom=self.custom_sampler_output)

            # DELAYED REJECTION
            if self.simulation_options.ntry > 1 and accept == 0:
                accept, new_set, outbound = self._sampling_methods.delayed_rejection.run_delayed_rejection(
                        old_set=self.__old_set,
                        new_set=new_set,
                        RDR=self._covariance._RDR,
                        ntry=self.simulation_options.ntry,
                        parameters=self.parameters,
                        invR=self._covariance._invR,
                        sosobj=self.__sos_object,
                        priorobj=self.__prior_object,
                        custom=self.custom_sampler_output)

            # UPDATE CHAIN & SUM-OF-SQUARES CHAIN
            self.__update_chain(accept=accept, new_set=new_set, outsidebounds=outbound)
            self.__sschain[self.__chain_index, :] = self.__old_set.ss

            # PRINT REJECTION STATISTICS
            if self.simulation_options.printint and iiprint == self.simulation_options.printint:
                print_rejection_statistics(
                        rejected=self.__rejected,
                        isimu=isimu,
                        iiadapt=iiadapt,
                        verbosity=self.simulation_options.verbosity)
                iiprint = 0  # reset print counter

            # ADAPTATION
            if self.simulation_options.adaptint > 0 and iiadapt == self.simulation_options.adaptint:
                self._covariance = self._sampling_methods.adaptation.run_adaptation(
                        covariance=self._covariance,
                        options=self.simulation_options,
                        isimu=isimu,
                        iiadapt=iiadapt,
                        rejected=self.__rejected,
                        chain=self.__chain,
                        chainind=self.__chain_index,
                        u=npar_sample_from_normal,
                        npar=self.parameters.npar,
                        alpha=new_set.alpha)

                iiadapt = 0  # reset local adaptation index
                self.__rejected['in_adaptation_interval'] = 0  # reset local rejection index

            # UPDATE ERROR VARIANCE
            if self.simulation_options.updatesigma:
                sigma2 = self._error_variance.update_error_variance(self.__old_set.ss, self.model_settings)
                self.__s2chain[self.__chain_index, :] = sigma2
                self.__old_set.sigma2 = sigma2

            # RUN CUSTOM SAMPLERS
            self.custom_sampler_output = []
            for cs in self.custom_samplers:
                self.custom_sampler_output.append(
                        cs.update(
                                accept=accept,
                                isimu=isimu,
                                current_set=self.__old_set))

            # SAVE TO LOG FILE
            if savecount == self.simulation_options.savesize:
                savecount, lastbin = self.__save_to_log_file(
                        chains=self.__chains,
                        start=isimu - self.simulation_options.savesize,
                        end=isimu)
                self.__save_to_log_file(
                        chains=[dict(mtx=np.dot(self._covariance._R.transpose(), self._covariance._R))],
                        start=isimu - self.simulation_options.savesize,
                        end=isimu,
                        append_to_log=False,
                        covmtx=True)
                # add custom chains is applicable
                for cs in self.custom_samplers:
                    if (hasattr(cs, 'save_chain') is True) and (cs.save_chain is True):
                        self.__save_to_log_file(
                                chains=cs.chains,
                                start=isimu - self.simulation_options.savesize,
                                end=isimu,
                                append_to_log=False)

        # SAVE REMAINING ELEMENTS TO BIN FILE
        self.__save_to_log_file(chains=self.__chains, start=lastbin, end=isimu + 1)
        self.__save_to_log_file(
                chains=[dict(mtx=np.dot(self._covariance._R.transpose(), self._covariance._R))],
                start=lastbin,
                end=isimu,
                append_to_log=False,
                covmtx=True)
        # add custom chains is applicable
        for cs in self.custom_samplers:
            if (hasattr(cs, 'save_chain') is True) and (cs.save_chain is True):
                self.__save_to_log_file(cs.chains, start=lastbin, end=isimu + 1, append_to_log=False)

        # update value to end value
        self.parameters._value[self.parameters._parind] = self.__old_set.theta

    # ------------------------------------------------
    def __generate_simulation_results(self):
        '''
        Generate simulation results dictionary.

        - Basic
        - Delayed rejection stats (if applicable)
        - Simulation options
        - Model settings
        - Sampling chain
        - Observation error chain
        - Sum-of-squares chain
        '''

        # BUILD RESULTS OBJECT
        self.simulation_results = ResultsStructure()  # inititialize
        self.simulation_results.add_basic(
                nsimu=self.simulation_options.nsimu,
                covariance=self._covariance,
                parameters=self.parameters,
                rejected=self.__rejected,
                simutime=self.__simulation_time,
                theta=self.parameters._value)

        if self.simulation_options.ntry > 1:
            self.simulation_results.add_dram(
                    drscale=self.simulation_options.drscale,
                    RDR=self._covariance._RDR,
                    total_rejected=self.__rejected['total'],
                    drsettings=self._sampling_methods.delayed_rejection)

        self.simulation_results.add_options(options=self.simulation_options)
        self.simulation_results.add_model(model=self.model_settings)

        # add prior information
        self.simulation_results.add_prior(
                mu=self.parameters._thetamu,
                sigma=self.parameters._thetasigma,
                priortype=self.model_settings.prior_type)
        # add chain, s2chain, and sschain
        self.simulation_results.add_chain(chain=self.__chain)
        self.simulation_results.add_s2chain(s2chain=self.__s2chain)
        self.simulation_results.add_sschain(sschain=self.__sschain)

    # ------------------------------------------------
    def __save_to_log_file(self, chains, start, end, append_to_log=True, covmtx=False):
        '''
        Save to log files

        Args:
            * **start** (:py:class:`int`): Start index of chain block to save
            * **end** (:py:class:`int`): End index of chain block to save

        Returns:
            * **savecount** (:py:class:`int`): Reset save counter
            * **lastbin** (:py:class:`int`): Last index saved
        '''
        if self.simulation_options.save_to_bin is True:
            savedir = self.simulation_options.savedir
            ChainProcessing._check_directory(savedir)
            if append_to_log is True:
                binlogfile = os.path.join(savedir, 'binlogfile.txt')
                binstr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ChainProcessing._add_to_log(binlogfile, str('{}\t{}\t{}\n'.format(binstr, start, end-1)))

            if covmtx is True:
                self.__save_covmtx_chain(chain=chains[0], start=start, end=end, extension='h5')
            else:
                self.__save_chains(chains=chains, savedir=savedir, start=start, end=end, extension='h5')

        if self.simulation_options.save_to_txt is True:
            savedir = self.simulation_options.savedir
            ChainProcessing._check_directory(savedir)
            if append_to_log is True:
                txtlogfile = os.path.join(savedir, 'txtlogfile.txt')
                txtstr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ChainProcessing._add_to_log(txtlogfile, str('{}\t{}\t{}\n'.format(txtstr, start, end-1)))
            if covmtx is True:
                self.__save_covmtx_chain(chain=chains[0], start=start, end=end, extension='txt')
            else:
                self.__save_chains(chains=chains, savedir=savedir, start=start, end=end, extension='txt')

        # reset counter
        savecount = 0
        lastbin = end
        return savecount, lastbin

    # --------------------------------------------------------
    def __save_chains(self, chains, savedir, start, end, extension):
        '''
        Save custom chain segment

        Args:
            * **chains** (:py:class:`list`): List of dicts with keys "file" and "mtx".  \
            "file" is name of log file, and "mtx" is chain array to save
            * **start** (:py:class:`int`): Starting index of chain to save to file
            * **end** (:py:class:`int`): Ending index of chain to save to file
            * **extension** (:py:class:`str`): File extension - 'h5' or 'txt'

        If you specify a `savesize` of 100, then every 100 simulations the last 100
        chain sets will be appended to the file.  That is to say, if you are on
        simulation 1000, the chain elements 900-999 will be appended to the file.
        '''

        for ii, chain in enumerate(chains):
            # add extension
            chainfile = ChainProcessing._create_path_with_extension(savedir, chain['file'], extension=extension)
            if extension.lower() == 'h5':
                # define set name based in start/end
                datasetname = str('{}_{}_{}'.format('nsimu', start, end-1))
                ChainProcessing._save_to_bin_file(chainfile, datasetname=datasetname, mtx=chain['mtx'][start:end, :])
            else:
                ChainProcessing._save_to_txt_file(chainfile, mtx=chain['mtx'][start:end, :])

    # --------------------------------------------------------
    def __save_covmtx_chain(self, chain, start, end, extension):
        '''
        Save custom chain segment

        Args:
            * **chains** (:py:class:`list`): List of dicts with keys "file" and "mtx".  \
            "file" is name of log file, and "mtx" is chain array to save
            * **start** (:py:class:`int`): Starting index of chain to save to file
            * **end** (:py:class:`int`): Ending index of chain to save to file
            * **extension** (:py:class:`str`): File extension - 'h5' or 'txt'

        If you specify a `savesize` of 100, then every 100 simulations the last 100
        chain sets will be appended to the file.  That is to say, if you are on
        simulation 1000, the chain elements 900-999 will be appended to the file.
        '''

        # add extension
        chainfile = ChainProcessing._create_path_with_extension(
                self.simulation_options.savedir,
                self.simulation_options.covchainfile,
                extension=extension)

        if extension.lower() == 'h5':
            # define set name based in start/end
            datasetname = str('{}_{}_{}'.format('nsimu', start, end-1))
            ChainProcessing._save_to_bin_file(chainfile, datasetname=datasetname, mtx=chain['mtx'])
        else:
            ChainProcessing._save_to_txt_file(chainfile, mtx=chain['mtx'])

    # --------------------------------------------------------
    def __update_chain(self, accept, new_set, outsidebounds):
        '''
        Update chain

        Args:
            * **accept** (:py:class:`str`): Flag to indicate whether :math:`q^*` is accepted or rejected
            * **new_set** (:class:`~.ParameterSet`): Features of :math:`q^*`
            * **outsidebounds** (:py:class:`bool`): Flag to indicate whether \
            rejection occured due to sampling outside limits
        '''
        if accept:
            # accept
            self.__chain[self.__chain_index, :] = new_set.theta
            self.__old_set = new_set
        else:
            # reject
            self.__chain[self.__chain_index, :] = self.__old_set.theta
            self.__update_rejected(outsidebounds=outsidebounds)

    # --------------------------------------------------------
    def __update_rejected(self, outsidebounds):
        '''
        Update rejection counters

        Args:
            * **outsidebounds** (:py:class:`bool`): Flag to indicate whether \
            rejection occured due to sampling outside limits
        '''
        self.__rejected['total'] += 1
        self.__rejected['in_adaptation_interval'] += 1
        if outsidebounds:
            self.__rejected['outside_bounds'] += 1

    # --------------------------------------------------------
    def display_current_mcmc_settings(self):
        '''
        Display model settings, simulation options, and current covariance values.

        Example display:

        ::

            model settings:
                sos_function = <function test_ssfun at 0x1c13c5d400>
                model_function = None
                sigma2 = [1.]
                N = [100.]
                N0 = [0.]
                S20 = [1.]
                nsos = 1
                nbatch = 1
            simulation options:
                nsimu = 5000
                adaptint = 100
                ntry = 2
                method = dram
                printint = 100
                lastadapt = 5000
                drscale = [5. 4. 3.]
                qcov = None
            covariance:
                qcov = [[0.01   0.    ]
                [0.     0.0625]]
                R = [[0.1  0.  ]
                [0.   0.25]]
                RDR = [array([[0.1 , 0.  ],
               [0.  , 0.25]]), array([[0.02, 0.  ],
               [0.  , 0.05]])]
                invR = [array([[10.,  0.],
               [ 0.,  4.]]), array([[50.,  0.],
               [ 0., 20.]])]
                last_index_since_adaptation = 0
                covchain = None
        '''
        self.model_settings.display_model_settings()
        self.simulation_options.display_simulation_options()
        self._covariance.display_covariance_settings()


# --------------------------------------------------------
def print_rejection_statistics(rejected, isimu, iiadapt, verbosity):
    '''
    Print Rejection Statistics.

    Threshold for printing is verbosity greater than or equal to 2.  If the rejection
    counters are as follows:

        - `total`: 144
        - `in_adaptation_interval`: 92
        - `outside_bounds`: 0

    Then we would expect the following display at the 200th simulation with an adaptation
    interval of 100.

    ::

        i:200 (72.0,92.0,0.0)

    Args:
        * **isimu** (:py:class:`int`): Simulation counter
        * **iiadapt** (:py:class:`int`): Adaptation counter
        * **verbosity** (:py:class:`int`): Verbosity of display output.
    '''
    message(verbosity, 2, str('i:{} ({},{},{})\n'.format(
            isimu, rejected['total']*isimu**(-1)*100, rejected['in_adaptation_interval']*iiadapt**(-1)*100,
            rejected['outside_bounds']*isimu**(-1)*100)))
