#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Wed Jan 17 09:08:13 2018

@author: prmiles
'''
# import required packages
import numpy as np
from datetime import datetime

class SimulationOptions:
    '''
    MCMC simulation options.

    Attributes:
       * :meth:`~define_simulation_options`
       * :meth:`~display_simulation_options`
    '''
    def __init__(self):
        # initialize simulation option variables
#        self.options = BaseSimulationOptions()
        self.description = 'simulation_options'
        self.__options_set = False

    def define_simulation_options(self, nsimu=int(1e4), adaptint = None, ntry = None, method='dram',
                 printint=None, adaptend = 0, lastadapt = 0, burnintime = 0,
                 waitbar = 1, debug = 0, qcov = None, updatesigma = False,
                 noadaptind = None, stats = 0, drscale = np.array([5, 4, 3], dtype = float),
                 adascale = None, savesize = 0, maxmem = 0, chainfile = 'chainfile',
                 s2chainfile = 's2chainfile', sschainfile = 'sschainfile', covchainfile = 'covchainfile', savedir = None,
                 save_to_bin = False, skip = 1, label = None, RDR = None, verbosity = 1, maxiter = None,
                 priorupdatestart = 0, qcov_adjust = 1e-8, burnin_scale = 10,
                 alphatarget = 0.234, etaparam = 0.7, initqcovn = None,
                 doram = None, rndseq = None, results_filename = None, save_to_json = False, save_to_txt = False,
                 json_restart_file = None):
        '''
        Define simulation options.

        Args:
            * **nsimu** (:py:class:`int`): Number of parameter samples to simulate.  Default is 1e4.
            * **adaptint** (:py:class:`int`): Number of interates between adaptation. Default is method dependent.
            * **ntry** (:py:class:`int`): Number of tries to take before rejection. Default is method dependent.
            * **method** (:py:class:`str`): Sampling method (:code:`'mh', 'am', 'dr', 'dram'`).  Default is :code:`'dram'`.
            * **printint** (:py:class:`int`): Printing interval.
            * **adaptend** (:py:class:`int`): Obsolete.
            * **lastadapt** (:py:class:`int`): Last adaptation iteration (i.e., no more adaptation beyond this point).
            * **burnintime** (:py:class:`int`):
            * **waitbar** (:py:class:`int`): Flag to use progress bar. Default is 1 -> on (otherwise -> off).
            * **debug** (:py:class:`int`): Flag to perform debug.  Default is 0 -> off.
            * **qcov** (:class:`~numpy.ndarray`): Proposal parameter covariance matrix.
            * **updatesigma** (:py:class:`bool`): Flag for updating measurement error variance. Default is 0 -> off (1 -> on).
            * **noadaptind** (:py:class:`int`): Indices not to be adapted in covariance matrix. Default is [] (untested).
            * **stats** (:py:class:`int`): Calculate convergence statistics. Default is 0 -> off (1 -> on).
            * **drscale** (:class:`~numpy.ndarray`): Reduced scale for sampling in DR algorithm. Default is [5,4,3].
            * **adascale** (:py:class:`float`): User defined covariance scale.  Default is method dependent (untested).
            * **savesize** (:py:class:`int`): Size of chain segments when saving to log files.  Default is 0.
            * **maxmem** (:py:class:`int`): Maximum memory available in mega bytes (Obsolete).
            * **chainfile** (:py:class:`str`): File name for :code:`chain` log file.
            * **sschainfile** (:py:class:`str`): File name for :code:`sschain` log file.
            * **s2chainfile** (:py:class:`str`): File name for :code:`s2chain` log file.
            * **covchainfile** (:py:class:`str`): File name for :code:`qcov` log file.
            * **savedir** (:py:class:`str`): Output directory of log files.  Default is current directory.
            * **save_to_bin** (:py:class:`bool`): Save log files to binary.  Default is False.
            * **save_to_txt** (:py:class:`bool`): Save log files to text.  Default is False.
            * **skip** (:py:class:`int`):
            * **label** (:py:class:`str`):
            * **RDR** (:class:`~numpy.ndarray`): R matrix for each stage of DR.
            * **verbosity** (:py:class:`int`): Verbosity of display output.
            * **maxiter** (:py:class:`int`): Obsolete.
            * **priorupdatestart**
            * **qcov_adjust** (:py:class:`float`): Adjustment scale for covariance matrix.
            * **burnin_scale** (:py:class:`float`): Scale for burnin.
            * **alphatarget** (:py:class:`float`): Acceptance ratio target.
            * **etaparam** (:py:class:`float`):
            * **initqcovn** (:py:class:`float`): Proposal covariance weight in update.
            * **doram** (:py:class:`int`): Flag to perform :code:`'ram'` algorithm (Obsolete).
            * **rndseq** (:class:`~numpy.ndarray`): Random number sequence (Obsolete).
            * **results_filename** (:py:class:`str`): Output file name when saving results structure with json.
            * **save_to_json** (:py:class:`bool`): Save results structure to json file.  Default is False.
            * **json_restart_file** (:py:class:`str`): Extract parameter covariance and last sample value from saved json file.

        .. note::

            For the log file names :code:`chainfile, sschainfile, s2chainfile` and :code:`covchainfile` do not include the extension.
            By specifying whether to save to text or to binary, the appropriate extension will be added.
        '''
        
        method_dictionary = {
            'mh': {'adaptint': 0, 'ntry': 1, 'doram': 0, 'adascale': adascale},
            'am': {'adaptint': 100, 'ntry': 1, 'doram': 0, 'adascale': adascale},
            'dr': {'adaptint': 0, 'ntry': 2, 'doram': 0, 'adascale': adascale},
            'dram': {'adaptint': 100, 'ntry': 2, 'doram': 0, 'adascale': adascale},
            'ram': {'adaptint': 1, 'ntry': 1, 'doram': 1, 'adascale': 1.},
            }
        
        # define items from dictionary
        if adaptint is None:
            self.adaptint = method_dictionary[method]['adaptint']  # update interval for adaptation
        elif method == 'mh' or method == 'dr':
            self.adaptint = method_dictionary[method]['adaptint']  # no adaptation - enforce!
        else:
            self.adaptint = adaptint
        
        if ntry is None:
            self.ntry = method_dictionary[method]['ntry']
        else:
            self.ntry = ntry
            
        if adascale is None:
            self.adascale = method_dictionary[method]['adascale']  # qcov_scale
        else:
            self.adascale = adascale
            
        if doram is None:
            self.doram = method_dictionary[method]['doram']
        else:
            self.doram = doram
        
        
        self.nsimu = nsimu  # length of chain to simulate
        self.method = method
        self.dodram = 0
        
        self.printint = printint  # print interval
        self.adaptend = adaptend  # last adapt
        self.lastadapt = lastadapt # last adapt
        self.burnintime = burnintime
        self.waitbar = waitbar # use waitbar
        self.debug = debug  # show some debug information
        self.qcov = qcov  # proposal covariance
        self.initqcovn = initqcovn  # proposal covariance weight in update
        self.updatesigma = updatesigma  #
        if noadaptind is None:
            self.noadaptind = [] # do not adapt these indices
        self.priorupdatestart = priorupdatestart
        self.qcov_adjust = qcov_adjust  # eps adjustment
        self.burnin_scale = burnin_scale
        self.alphatarget = alphatarget  # acceptance ratio target
        self.etaparam = etaparam  #
        self.stats = stats  # convergence statistics
        self.drscale = drscale

        self.skip = skip
        
        datestr = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.datestr = datestr
        
        if label is None:
            self.label = str('MCMC run at {}'.format(datestr))
        else:
            self.label = label
            
        self.RDR = RDR
        self.verbosity = verbosity # amount of information to print
        self.maxiter = maxiter
        
        # log settings
        self.savesize = savesize
        self.maxmem = maxmem
            
        self.chainfile = chainfile
        self.s2chainfile = s2chainfile
        self.sschainfile = sschainfile
        self.covchainfile = covchainfile
        
        if savedir is None:
            self.savedir = str('{}_{}'.format(datestr,'chain_log'))
        else:
            self.savedir = savedir
            
        self.save_to_bin = save_to_bin
        self.save_to_txt = save_to_txt
        
        self.results_filename = results_filename
        self.save_to_json = save_to_json
        self.json_restart_file = json_restart_file
        
        self.__options_set = True # options have been defined
        
    def _check_dependent_simulation_options(self, model):
        '''
        Check dependent parameters.
        - Checks that :code:`savesize` is between 0 and :code:`nsimu`.
        - If :code:`ntry` greater than 1, then assume delayed rejection was wanted.
        - If :code:`lastadapt` less than 1, then set equal to :code:`nsimu`.
        - Update :code:`printint` based on size of :code:`adaptint`.
        - If :code:`N0` not None, turn on :code:`updatesigma`.

        Args:
            * **model**: (:class:`~.ModelSettings`): MCMC model settings.
        '''
        # save options
        if self.savesize <= 0 or self.savesize > self.nsimu:
            self.savesize = self.nsimu
        
        # turn on DR if ntry > 1
        if self.ntry > 1:
            self.dodram = 1
        else:
            self.dodram = 0
            
        if self.lastadapt < 1:
            self.lastadapt = self.nsimu
            
        if self.printint is None:
            self.printint = max(100,min(1000,self.adaptint))
            
        # if N0 given, then also turn on updatesigma
        if model.N0 is not None:
            self.updatesigma = True
            
    def display_simulation_options(self, print_these = None):
        '''
        Display subset of the simulation options.
        
        Args:
            * **print_these** (:py:class:`list`): List of strings corresponding to keywords.  Default below.

        ::

            print_these = ['nsimu', 'adaptint', 'ntry', 'method', 'printint', 'lastadapt', 'drscale', 'qcov']
        '''
        if print_these is None:
            print_these = ['nsimu', 'adaptint', 'ntry', 'method', 'printint', 'lastadapt', 'drscale', 'qcov']
            
        print('simulation options:')
        for ptii in print_these:
            print('\t{} = {}'.format(ptii, getattr(self, ptii)))
            
        return print_these