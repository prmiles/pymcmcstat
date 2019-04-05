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
from ..utilities.general import removekey
from ..chain.ChainProcessing import _check_directory, _create_path_without_extension


class ResultsStructure:
    '''
    Results from MCMC simulation.

    **Description:** Class used to organize results of MCMC simulation.

    Attributes:
        * :meth:`~export_simulation_results_to_json_file`
        * :meth:`~determine_filename`
        * :meth:`~save_json_object`
        * :meth:`~load_json_object`
        * :meth:`~add_basic`
        * :meth:`~add_updatesigma`
        * :meth:`~add_dram`
        * :meth:`~add_prior`
        * :meth:`~add_options`
        * :meth:`~add_model`
        * :meth:`~add_chain`
        * :meth:`~add_s2chain`
        * :meth:`~add_sschain`
        * :meth:`~add_time_stats`
        * :meth:`~add_random_number_sequence`
    '''
    def __init__(self):
        self.results = {}  # initialize empty dictionary
        self.basic = False  # basic structure not add yet

    # --------------------------------------------------------
    def export_simulation_results_to_json_file(self, results):
        '''
        Export simulation results to a json file.

        Args:
            * **results** (:class:`~.ResultsStructure`): Dictionary of MCMC simulation results/settings.
        '''
        savedir = results['simulation_options']['savedir']
        filename = self.determine_filename(options=results['simulation_options'])
        _check_directory(savedir)  # make sure output directory exists
        file = _create_path_without_extension(savedir, filename)
        self.save_json_object(results, file)

    @classmethod
    def determine_filename(cls, options):
        '''
        Determine results filename.

        If not specified by `results_filename` in the simulation options, then
        a default naming format is generated using the date string associated
        with the initialization of the simulation.

        Args:
            * **options** (:class:`~.SimulationOptions`): MCMC simulation options.

        Returns:
            * **filename** (:py:class:`str`): Filename string.
        '''
        results_filename = options['results_filename']
        if results_filename is None:
            dtstr = options['datestr']
            filename = str('{}{}{}'.format(dtstr, '_', 'mcmc_simulation.json'))
        else:
            filename = results_filename
        return filename

    @classmethod
    def save_json_object(cls, results, filename):
        '''
        Save object to json file.

        .. note::

            Filename should include extension.

        Args:
            * **results** (:py:class:`dict`): Object to save.
            * **filename** (:py:class:`str`): Write object into file with this name.
        '''
        with open(filename, 'w') as out:
            json.dump(results, out, sort_keys=True, indent=4, cls=NumpyEncoder)

    @classmethod
    def load_json_object(cls, filename):
        '''
        Load object stored in json file.

        .. note::

            Filename should include extension.

        Args:
            * **filename** (:py:class:`str`): Load object from file with this name.

        Returns:
            * **results** (:py:class:`dict`): Object loaded from file.
        '''
        with open(filename, 'r') as obj:
            results = json.load(obj)
        return results

    # --------------------------------------------------------
    def add_basic(self, nsimu, covariance, parameters, rejected, simutime, theta):
        '''
        Add basic results from MCMC simulation to structure.

        Args:
            * **nsimu** (:py:class:`int`): Number of MCMC simulations.
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
        self.results['total_rejected'] = rejected['total']*(nsimu**(-1))  # total rejected
        # rejected due to sampling outside limits
        self.results['rejected_outside_bounds'] = rejected['outside_bounds']*(nsimu**(-1))
        self.results['R'] = covariance._R
        self.results['qcov'] = np.dot(covariance._R.transpose(), covariance._R)
        self.results['cov'] = covariance._covchain
        self.results['qcov_scale'] = covariance._qcov_scale
        self.results['mean'] = covariance._meanchain
        self.results['names'] = [parameters._names[ii] for ii in parameters._parind]
        self.results['limits'] = [parameters._lower_limits[parameters._parind[:]],
                                  parameters._upper_limits[parameters._parind[:]]]
        self.results['nsimu'] = nsimu
        self.results['simutime'] = simutime
        covariance._qcovorig[np.ix_(parameters._parind, parameters._parind)] = self.results['qcov']
        self.results['qcovorig'] = covariance._qcovorig
        self.results['original_covariance'] = covariance._qcov_original
        self.basic = True  # add_basic has been execute

    def add_updatesigma(self, updatesigma, sigma2, S20, N0):
        '''
        Add information to results structure related to observation error.

        Args:
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

    def add_dram(self, drscale, RDR, total_rejected, drsettings):
        '''
        Add results specific to performing DR algorithm.

        Args:
            * **drscale** (:class:`~numpy.ndarray`): Reduced scale for sampling in DR algorithm. Default is [5,4,3].
            * **RDR** (:class:`~numpy.ndarray`): Cholesky decomposition of covariance matrix based on DR.
            * **total_rejected** (:py:class:`int`): Number of rejected samples.
            * **drsettings** (:class:`~.DelayedRejection`): Need access to counters within DR class.
        '''
        # extract results from basic structure
        if self.basic is True:
            nsimu = self.results['nsimu']
            self.results['drscale'] = drscale
            drsettings.iacce[0] = nsimu - total_rejected - sum(drsettings.iacce[1:])
            # 1 - number accepted without DR, 2 - number accepted via DR try 1,
            # 3 - number accepted via DR try 2, etc.
            self.results['iacce'] = drsettings.iacce
            self.results['alpha_count'] = drsettings.dr_step_counter
            self.results['RDR'] = RDR
            return True
        else:
            print('Cannot add DRAM settings to results structure before running ''add_basic''')
            return False

    def add_prior(self, mu, sigma, priortype):
        '''
        Add results specific to prior function.

        Args:
            * **mu** (:class:`~numpy.ndarray`): Prior mean.
            * **sigma** (:class:`~numpy.ndarray`): Prior standard deviation.
            * **priortype** (:py:class:`int`): Flag identifying type of prior.

        .. note::

            This feature is not currently implemented.
        '''
        self.results['prior'] = dict(mu=mu, sigma=sigma, priortype=priortype)

    def add_options(self, options=None):
        '''
        Saves subset of features of the simulation options in a nested dictionary.

        Args:
            * **options** (:class:`.SimulationOptions`): MCMC simulation options.
        '''
        # Return options as dictionary
        opt = options.__dict__
        # define list of keywords to NOT add to results structure
        do_not_save_these_keys = ['doram', 'waitbar', 'debug', 'dodram', 'maxmem',
                                  'verbosity', 'RDR', 'stats', 'initqcovn', 'drscale',
                                  'maxiter', '_SimulationOptions__options_set', 'skip']
        for keyii in do_not_save_these_keys:
            opt = removekey(opt, keyii)
        # must convert 'options' object to a dictionary
        self.results['simulation_options'] = opt

    def add_model(self, model=None):
        '''
        Saves subset of features of the model settings in a nested dictionary.

        Args:
            * **model** (:class:`.ModelSettings`): MCMC model settings.
        '''
        # Return model as dictionary
        mod = model.__dict__
        # define list of keywords to NOT add to results structure
        do_not_save_these_keys = ['sos_function', 'prior_function', 'model_function',
                                  'prior_update_function', 'prior_pars']
        for keyii in do_not_save_these_keys:
            mod = removekey(mod, keyii)
        # must convert 'model' object to a dictionary
        self.results['model_settings'] = mod

    def add_chain(self, chain=None):
        '''
        Add chain to results structure.

        Args:
            * **chain** (:class:`~numpy.ndarray`): Model parameter sampling chain.
        '''
        self.results['chain'] = chain

    def add_s2chain(self, s2chain=None):
        '''
        Add observiation error chain to results structure.

        Args:
            * **s2chain** (:class:`~numpy.ndarray`): Sampling chain of observation errors.
        '''
        self.results['s2chain'] = s2chain

    def add_sschain(self, sschain=None):
        '''
        Add sum-of-squares chain to results structure.

        Args:
            * **sschain** (:class:`~numpy.ndarray`): Calculated sum-of-squares error for each parameter chains set.
        '''
        self.results['sschain'] = sschain

    def add_time_stats(self, mtime, drtime, adtime):
        '''
        Add time spend using each sampling algorithm.

        Args:
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

        Args:
            * **rndseq** (:class:`~numpy.ndarray`): Sequence of sampled random numbers.

        .. note::

            This feature is not currently implemented.
        '''
        self.results['rndseq'] = rndseq
