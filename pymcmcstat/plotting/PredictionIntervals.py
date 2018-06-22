#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:00:11 2017

@author: prmiles
"""

import numpy as np
import sys
from scipy.interpolate import interp1d
from ..settings.DataStructure import DataStructure
from ..settings.ModelSettings import ModelSettings
from ..utilities.progressbar import progress_bar
from ..plotting.utilities import append_to_nrow_ncol_based_on_shape, convert_flag_to_boolean, set_local_parameters
import matplotlib.pyplot as plt

class PredictionIntervals:
    '''
    Prediction/Credible interval methods.
    
    :Attributes:
        - :meth:`~setup_prediction_interval_calculation`
        - :meth:`~generate_prediction_intervals`
        - :meth:`~plot_prediction_intervals`
    '''
    # --------------------------------------------
    def setup_prediction_interval_calculation(self, results, data, modelfunction):
        '''
        Setup calculation for prediction interval generation
        
        :Args:
            * results (:class:`~.ResultsStructure`): MCMC results structure
            * data (:class:`~.DataStructure`): MCMC data structure
            * modelfunction: Model function handle
            
        '''
        # Analyze data structure
        self.__ndatabatches, nrows, ncols = self._analyze_data_structure(data = data)
            
        # setup data structure for prediction
        self.datapred = self._setup_data_structure_for_prediction(data = data, ndatabatches = self.__ndatabatches)
            
        # assign model function
        self.modelfunction = modelfunction
        
        # assign required features from the results structure
        self._assign_features_from_results_structure(results = results)
        
        # evaluate model function to determine shape of response
        self.__nrow, self.__ncol = self._determine_shape_of_response(modelfunction = modelfunction, ndatabatches = self.__ndatabatches, datapred = self.datapred, theta = self.__theta)
        
        # analyze structure of s2chain with respect to model output
        if self.__s2chain is not None:
            self.__s2chain_index = self._analyze_s2chain(ndatabatches = self.__ndatabatches, s2chain = self.__s2chain, ncol = self.__ncol)
    # -------------------------------------------- 
    @classmethod
    def _analyze_data_structure(cls, data):
        '''
        Analyze data structure.
        
        :Args:
            * **data** (:class:`~.DataStructure`): Data structure.
            
        :Returns:
            * **ndatabatches** (:py:class:`int`): Number of batch data sets.
            * **nrows** (:py:class:`list`): List containing number of rows in each set.
            * **ncols** (:py:class:`list`): List containing number of columns in each set.
        '''
        # Analyze data structure
        dshapes = data.shape
        ndatabatches = len(dshapes)
        nrows = []
        ncols = []
        for ii in range(ndatabatches):
            nrows.append(dshapes[ii][0])
            if len(dshapes[ii]) != 1:
                ncols.append(dshapes[ii][1])
            else:
                ncols.append(1)
        return ndatabatches, nrows, ncols
    # --------------------------------------------
    @classmethod
    def _setup_data_structure_for_prediction(cls, data, ndatabatches):
        '''
        Setup data structure for generating quantile.
        
        :Args:
            * **data** (:class:`~.DataStructure`): Data structure.
            * **ndatabatches** (:py:class:`int`): Number of batch data sets.
            
        :Returns:
            * **datapred** (:py:class:`list`): Data structure for interval generation.
        '''
        datapred = []
        for ii in range(ndatabatches):
            # setup data structure for prediction
            # this is required to allow user to send objects other than xdata to model function
            datapred.append(DataStructure())
            datapred[ii].add_data_set(x = data.xdata[ii], y = data.ydata[ii],
                         user_defined_object = data.user_defined_object[ii])
            
        return datapred
    # --------------------------------------------
    def _assign_features_from_results_structure(self, results):
        '''
        Define variables based on items extracted from results dictionary.
        
        :Args:
            * **results** (:py:class:`dict`): Results dictionary from MCMC simulation.
        '''
        # assign required features from the results structure
        self.__chain = results['chain']
        self.__s2chain = results['s2chain']
        self.__parind = results['parind']
        self.__local = results['local']
        # Check how 'model' object was saved in results structure
        if isinstance(results['model_settings'], ModelSettings):
            self.__nbatch = results['model'].nbatch
        else:
            self.__nbatch = results['model_settings']['nbatch']
            
        self.__theta = results['theta']
        
        if 'sstype' in results:
            self.__sstype = results['sstype']
        else:
            self.__sstype = 0
    # --------------------------------------------
    @classmethod      
    def _determine_shape_of_response(cls, modelfunction, ndatabatches, datapred, theta):
        '''
        Determine shape of model function repsonse.
        
        :Args:
            * **modelfun** (:py:class:`func`): Model function handle.
        '''
        # evaluate model function to determine shape of response
        nrow = []
        ncol = []
        for ii in range(ndatabatches):
            if isinstance(modelfunction, list):
                y = modelfunction[ii](datapred[ii], theta)
            else:
                y = modelfunction(datapred[ii], theta)
                
            sh = y.shape
            nrow, ncol = append_to_nrow_ncol_based_on_shape(sh, nrow, ncol)
        return nrow, ncol

    # --------------------------------------------
    @classmethod
    def _analyze_s2chain(cls, ndatabatches, s2chain, ncol):
        '''
        Analysis of s2chain.
        
        Depending on the nature of the data structure, there are a couple ways
        to interpret the shape of the s2chain.  If s2chain = [nsimu, ns2], then
        ns2 could correspond to len(data.ydata) or the number of columns in
        data.ydata[0].  We will assume in this code that the user either stores
        similar sized vectors in a matrix; or, creates distinct list elements for
        equal or different size vectors.  Creating distinct list elements of matrices
        will generate an error when trying to plot prediction intervals.
        
        :Args:
            * **ndatabatches** (:py:class:`int`): Number of batch data sets.
        '''
        # shape of s2chain
        n = s2chain.shape[1]
        
        # compare shape of s2chain with the number of data batches and the
        # number of columns in each batch.
        total_columns = sum(ncol)
        
        if n == 1: # only one obs. error for all data sets
            s2chain_index = np.zeros([ndatabatches,2], dtype = int)
            for ii in range(ndatabatches):
                s2chain_index[ii,:] = np.array([0, 1])
            
        elif n != 1 and total_columns == n: # then different obs. error for each column
            s2chain_index = np.zeros([ndatabatches,2], dtype = int)
            for ii in range(ndatabatches):
                if ii == 0:
                    s2chain_index[ii,:] = np.array([0, ncol[ii]])
                else:
                    s2chain_index[ii,:] = np.array([s2chain_index[ii-1,1], s2chain_index[ii-1,1] + ncol[ii]])
        elif n != 1 and total_columns != n:
            if n == ndatabatches: # assume separate obs. error for each batch
                s2chain_index = np.zeros([ndatabatches,2], dtype = int)
                for ii in range(ndatabatches):
                    if ii == 0: # 1?
                        s2chain_index[ii,:] = np.array([0, 1])
                    else:
                        s2chain_index[ii,:] = np.array([s2chain_index[ii-1,1], s2chain_index[ii-1,1] + 1])
            else:
                print('s2chain.shape = {}'.format(s2chain.shape))
                print('ndatabatches = {}'.format(ndatabatches))
                print('# of columns per batch = {}'.format(ncol))
                sys.exit('Unclear data structure: error variances do not match size of model output')
                
        return s2chain_index
        
    # --------------------------------------------
    def generate_prediction_intervals(self, sstype = None, nsample = 500, calc_pred_int = True, waitbar = False):
        '''
        Generate prediction/credible interval.
        
        :Args:
            * **sstype** (:py:class:`int`): Sum-of-squares type
            * **nsample** (:py:class:`int`): Number of samples to use in generating intervals.
            * **calc_pred_int** (:py:class:`bool`): Flag to turn on prediction interval calculation.
            * **waitbar** (:py:class:`bool`): Flag to turn on progress bar.
        '''
        
        chain, s2chain, nsimu, lims, sstype, nsample, iisample = self._setup_generation_requirements(sstype = sstype, nsample = nsample, calc_pred_int = calc_pred_int)
        
        # setup progress bar
        print('Generating credible/prediction intervals:\n')
        if waitbar is True:
            self.__wbarstatus = progress_bar(iters = int(nsample))
            
        # extract chain elements
        testchain = chain[iisample,:]
            
        # loop through data sets
        credible_intervals = []
        prediction_intervals = []
        for ii in range(len(self.datapred)):
            datapredii, nrow, ncol, modelfun, test = self._setup_predii(ii = ii, datapred = self.datapred, nrow = self.__nrow, ncol = self.__ncol, modelfunction = self.modelfunction, local = self.__local)
            if s2chain is not None:
                s2ci = [self.__s2chain_index[ii][0], self.__s2chain_index[ii][1]]
                tests2chain = s2chain[iisample, s2ci[0]:s2ci[1]]
            else:
                tests2chain = None
            
            # Run interval generation on set ii
            ysave, osave = self._run_predii(testchain, tests2chain, nrow, ncol, waitbar, sstype, test, modelfun, datapredii)
                
            # generate quantiles
            plim, olim = self._generate_quantiles(ysave, osave, lims, ncol, s2chain)
                
            credible_intervals.append(plim)
            prediction_intervals.append(olim)
            
        if s2chain is None:
            prediction_intervals = None
            
        # generate output dictionary
        self.intervals = {'credible_intervals': credible_intervals,
               'prediction_intervals': prediction_intervals}
    
        print('\nInterval generation complete\n')
    # --------------------------------------------
    @classmethod
    def _setup_predii(cls, ii, datapred, nrow, ncol, modelfunction, local):
        '''
        Setup value for interval ii.
        
        :Args:
            * **ii** (:py:class:`int`): Iteration number.
            
        :Returns:
            * **datapredii** (:class:`~numpy.ndarray`): Data set.
            * **nrow** (:py:class:`int`): Number of rows in data set.
            * **ncol** (:py:class:`int`): Number of columns in data set.
            * **modelfun** (:py:class:`func`): Model function handle.
            * **test** (:class:`~numpy.ndarray`): Array of booleans correponding to local test.
        '''
        datapredii = datapred[ii]
        nrow = nrow[ii]
        ncol = ncol[ii]
        if isinstance(modelfunction, list):
            modelfun = modelfunction[ii]
        else:
            modelfun = modelfunction
        
        # some parameters may only apply to certain batch sets
        test = set_local_parameters(ii = ii, local = local)
        return datapredii, nrow, ncol, modelfun, test
    

    # --------------------------------------------
    def _run_predii(self, testchain, tests2chain, nrow, ncol, waitbar, sstype, test, modelfun, datapredii):
        '''
        Calculate observations for set ii.
        
        :Args:
            * **testchain** (:class:`~numpy.ndarray`): Sample points from posterior density.
            * **s2chain** (:class:`~numpy.ndarray`): Sample points from observation errors.
            * **nrow** (:py:class:`int`): Number of rows in data set.
            * **ncol** (:py:class:`int`): Number of columns in data set.
            * **waitbar** (:py:class:`bool`): Flag to turn on progress bar.
            * **sstype** (:py:class:`int`): Flag to specify sstype.
            * **test** (:class:`~numpy.ndarray`): Array of booleans correponding to local test.
            * **modelfun** (:py:class:`func`): Model function handle.
            * **datapredii** (:class:`~numpy.ndarray`): Data set.
            
        :Returns:
            * **ysave** (:class:`~numpy.ndarray`): Model responses.
            * **osave** (:class:`~numpy.ndarray`): Model responses with observation errors.
        '''
        nsample, npar = testchain.shape
        theta = self.__theta
        ysave = np.zeros([nsample, nrow, ncol])
        osave = np.zeros([nsample, nrow, ncol])
        
        for kk, isa in enumerate(testchain):
            # progress bar
            if waitbar is True:
                self.__wbarstatus.update(kk)
            
            # extract chain set
            theta[self.__parind[:]] = isa
            th = theta[test]
            # evaluate model
            ypred = modelfun(datapredii, th)
            ypred = ypred.reshape(nrow, ncol)
            
            if tests2chain is not None:
                s2elem = tests2chain[kk]
                if s2elem.shape != (1,s2elem.size):
                    s2elem = s2elem.reshape(1,s2elem.shape[0]) # make row vector
                opred = self._observation_sample(s2elem, ypred, sstype)
            else:
                opred = np.zeros([nrow, ncol])
               
            # store model prediction
            ysave[kk,:,:] = ypred # store model output
            osave[kk,:,:] = opred # store model output with observation errors
        return ysave, osave
    # --------------------------------------------
    def _generate_quantiles(self, ysave, osave, lims, ncol, s2chain):
        '''
        Generate quantiles based on observations.
        
        :Args:
            * **ysave** (:class:`~numpy.ndarray`): Array of model responses.
            * **osave** (:class:`~numpy.ndarray`): Array of model responses plus observation errors.
            * **lims** (:class:`~numpy.ndarray`): Array of quantile limits.
            * **ncol** (:py:class:`int`): Number of columns in `ysave`.
            * **s2chain** (:class:`~numpy.ndarray`): Array of observation errors.
            
        :Returns:
            * **credible_quantiles** (:py:class:`list`): Quantiles for credible intervals.
            * **prediction_quantiles** (:py:class:`list`): Quantiles for prediction intervals
        '''
        # generate quantiles
        credible_quantiles = []
        prediction_quantiles = []
        for jj in range(ncol):
            credible_quantiles.append(self._empirical_quantiles(ysave[:,:,jj], lims))
            if s2chain is not None:
                prediction_quantiles.append(self._empirical_quantiles(osave[:,:,jj], lims))
            
        return credible_quantiles, prediction_quantiles
    # --------------------------------------------
    def _setup_generation_requirements(self, nsample, calc_pred_int, sstype):
        '''
        Check if number of samples is defined - default to number of simulations.
        
        :Args:
            * **nsample** (:py:class:`int`): Number of samples to draw from posterior.
            * **calc_pred_int** (:py:class:`bool`): User defined flag as to whether or not they want PI plotted.
            * **sstype** (:py:class:`int`): Flag to specify sstype.
            
        :Returns:
            * **chain** (:class:`~numpy.ndarray`): Chain of posterior density.
            * **s2chain** (:class:`~numpy.ndarray`): Chain of observation errors.
            * **nsimu** (:py:class:`int`): Number of MCMC simulations.
            * **lims** (:class:`~numpy.ndarray`): Quantile limits.
            * **sstype** (:py:class:`int`): Flag to specify sstype.
            * **nsample** (:py:class:`int`): Number of samples to draw from posterior.
            * **iisample** (:class:`~numpy.ndarray`): Array of indices in posterior set.
        '''
        # extract chain & s2chain from results
        chain = self.__chain
        
        calc_pred_int = convert_flag_to_boolean(calc_pred_int)
        if calc_pred_int is False:
            s2chain = None
        else:
            s2chain = self.__s2chain
        
        # define number of simulations by the size of the chain array
        nsimu = chain.shape[0]
        
        # define interval limits
        lims = self._setup_interval_limits(s2chain)
        
        # define ss type
        sstype = self._setup_sstype(sstype)
        
        # check value of nsample
        nsample = self._check_nsample(nsample = nsample, nsimu = nsimu)
            
        # define sample points
        iisample, nsample = self._define_sample_points(nsample = nsample, nsimu = nsimu)
        
        return chain, s2chain, nsimu, lims, sstype, nsample, iisample
    # --------------------------------------------
    @classmethod
    def _setup_interval_limits(cls, s2chain):
        '''
        Setup interval limits.
        
        If the observation errors are `None`, then it means prediction intervals
        are irrelevant.  As a default the routine will then plot an expanded
        set of credible intervals.  The output of this function should be
        interpreted as lower/upper limits of quantiles.  So, a 99% quantile will
        have the limits [0.005, ..., 0.995].
        
        :Args:
            * **sigma2** (:class:`~numpy.ndarray` or `None`): Observation error chain.
            
        \\
        
        :Returns:
            * **lims** (:class:`~numpy.ndarray`): Lower/Upper limits of quantiles.
        '''
        if s2chain is None:
            lims = np.array([0.005,0.025,0.05,0.25,0.5,0.75,0.9,0.975,0.995])
        else:
            lims = np.array([0.025, 0.5, 0.975])
        return lims 
    # --------------------------------------------
    def _setup_sstype(self, sstype):
        '''
        Setup sstype and check value.
        
        Cases:
            - `sstype = 0`: Standard normal.
            - `sstype = 1`: Square root.
            - `sstype = 2`: Logarithmic.
            
        See example usage in :meth:`~._observation_sample`.
        
        :Args:
            * **sstype** (:py:class:`int`): Flag to specify sstype.
            
        :Returns:
            * **sstype** (:py:class:`int`): Flag to specify sstype
        '''
        if sstype is None:
            sstype = self.__sstype
        else:
            sstype = 0
        return sstype
    # --------------------------------------------
    @classmethod
    def _check_nsample(cls, nsample, nsimu):
        '''
        Check if number of samples is defined - default to number of simulations.
        
        :Args:
            * **nsample** (:py:class:`int`): Number of samples to draw from posterior.
            * **nsimu** (:py:class:`int`): Number of MCMC simulations.
            
        :Returns:
            * **nsample** (:py:class:`int`): Number of samples to draw from posterior.
        '''
        # check value of nsample
        if nsample is None:
            nsample = nsimu
        return nsample
    # --------------------------------------------
    @classmethod
    def _define_sample_points(cls, nsample, nsimu):
        '''
        Define indices to sample from posteriors.
        
        :Args:
            * **nsample** (:py:class:`int`): Number of samples to draw from posterior.
            * **nsimu** (:py:class:`int`): Number of MCMC simulations.
            
        :Returns:
            * **iisample** (:class:`~numpy.ndarray`): Array of indices in posterior set.
            * **nsample** (:py:class:`int`): Number of samples to draw from posterior.
        '''
        # define sample points
        if nsample >= nsimu:
            iisample = range(nsimu) # sample all points from chain
            nsample = nsimu
        else:
            # randomly sample from chain
            iisample = np.ceil(np.random.rand(nsample)*nsimu) - 1
            iisample = iisample.astype(int)
        return iisample, nsample

    # --------------------------------------------
    def plot_prediction_intervals(self, plot_pred_int = True, adddata = False, addlegend = True, figsizeinches = None):
        '''
        Plot prediction/credible intervals.
        
        :Args:
            * **plot_pred_int** (:py:class:`bool`): Flag to include PI on plot.
            * **adddata** (:py:class:`bool`): Flag to include data on plot.
            * **addlegend** (:py:class:`bool`): Flag to include legend on plot.
            * **figsizeinches** (:py:class:`list`): Specify figure size in inches [Width, Height].
        '''
        # unpack dictionary
        credible_intervals = self.intervals['credible_intervals']
        prediction_intervals = self.intervals['prediction_intervals']
        
        prediction_intervals, figsizeinches, nbatch, nn, clabels, plabels = self._setup_interval_plotting(
                plot_pred_int, prediction_intervals, credible_intervals, figsizeinches)
        
        # initialize figure handle
        fighandle = []
        axhandle = []
        for ii in range(self.__ndatabatches):
                          
            credlims = credible_intervals[ii] # should be ny lists inside
            ny = len(credlims)
            
            # extract data
            dataii = self.datapred[ii]
            
            # define independent data
            time = dataii.xdata[0].reshape(dataii.xdata[0].shape[0],)
            
            for jj in range(ny):
                htmp, ax = self._initialize_plot_features(ii = ii, jj = jj, ny = ny, figsizeinches = figsizeinches)
                fighandle.append(htmp)
                axhandle.append(ax)

                intcol = [0.85, 0.85, 0.85] # dimmest (lightest) color
                
                # add prediction intervals - if applicable
                if prediction_intervals is not None:
                    ax.fill_between(time, prediction_intervals[ii][jj][0], prediction_intervals[ii][jj][-1],
                                    facecolor = intcol, alpha = 0.5, label = plabels[0])
                    intcol = [0.75, 0.75, 0.75]
                
                # add first credible interval
                ax.fill_between(time, credlims[jj][0], credlims[jj][-1], facecolor = intcol, alpha = 0.5, label = clabels[0])
                
                # add range of credible intervals - if applicable
                for kk in range(1,int(nn)-1):
                    tmpintcol = np.array(intcol)*0.9**(kk)
                    ax.fill_between(time, credlims[jj][kk], credlims[jj][-kk - 1], facecolor = tmpintcol, alpha = 0.5, label = clabels[kk])
                    
                # add model (median parameter values)
                ax.plot(time, credlims[jj][int(nn)-1], '-k', linewidth=2, label = 'model')
                
                # add data to plot
                if adddata is True:
                    plt.plot(dataii.xdata[0], dataii.ydata[0][:,jj], '.b', linewidth=1, label = 'data')
                    
                # add title
                self._add_batch_column_title(nbatch, ny, ii, jj)
                    
                # add legend
                if addlegend is True:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc='upper left')
    
        return fighandle, axhandle
    
    @classmethod
    def _initialize_plot_features(cls, ii, jj, ny, figsizeinches):
        '''
        Initialize plot for prediction/credible intervals.
        
        :Args:
            * **ii** (:py:class:`int`): Batch #
            * **jj** (:py:class:`int`): Column #
            * **ny** (:py:class:`ny`): Number of intervals
            * **figsizeinches** (:py:class:`list`): Specify figure size in inches [Width, Height].
        '''
        fighandbatch = str('Batch # {}'.format(ii))
        fighandcolumn = str('Column # {}'.format(jj))
        fighand = str('{} | {}'.format(fighandbatch, fighandcolumn))
        htmp = plt.figure(fighand, figsize=(figsizeinches)) # create new figure
        
        plt.figure(fighand)
        ax = plt.subplot(ny,1,jj+1)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        
        return htmp, ax
     
    @classmethod
    def _add_batch_column_title(cls, nbatch, ny, ii, jj):
        '''
        Add title to plot based on batch/column number.
        
        :Args:
            * **nbatch** (:py:class:`nbatch`): Number of batches
            * **ny** (:py:class:`ny`): Number of intervals
            * **ii** (:py:class:`int`): Batch #
            * **jj** (:py:class:`int`): Column #
        '''
        # add title
        if nbatch > 1:
            plt.title(str('Batch #{}, Column #{}'.format(ii,jj)))
        elif ny > 1:
            plt.title(str('Column #{}'.format(jj)))
                          
    # --------------------------------------------
    def _setup_interval_plotting(self, plot_pred_int, prediction_intervals, credible_intervals, figsizeinches):
        '''
        Setup variables for interval plotting
        
        :Args:
            * **plot_pred_int** (:py:class:`bool`): Flag to plot prediction interval
            * **prediction_intervals** (:py:class:`list`): Prediction intervals
            * **credible_intervals** (:py:class:`list`): Credible intervals
            * **figsizeinches** (:py:class:`list`): Specify figure size in inches [Width, Height].
            
        :Returns:
            * **prediction_intervals** (:py:class:`list` or `None`): Prediction intervals
            * **figsizeinches** (:py:class:`list`): Specify figure size in inches [Width, Height].
            * **nbatch** (:py:class:`int`): Number of batches
            * **nn** (:py:class:`int`): Line number corresponding to median.
            * **clabels** (:py:class:`list`): List of label strings for credible intervals.
            * **plabels** (:py:class:`list`): List of label strings for prediction intervals.
        '''
        prediction_intervals = self._check_prediction_interval_flag(plot_pred_int = plot_pred_int, prediction_intervals = prediction_intervals)
            
        # check if figure size was specified
        if figsizeinches is None:
            figsizeinches = [7,5]
            
        nbatch, nn, nlines = self._setup_counting_metrics(credible_intervals = credible_intervals)
        
        clabels, plabels = self._setup_labels(prediction_intervals = prediction_intervals, nlines = nlines)
        return prediction_intervals, figsizeinches, nbatch, nn, clabels, plabels
    
    # --------------------------------------------
    def _check_prediction_interval_flag(self, plot_pred_int, prediction_intervals):
        '''
        Check prediction interval flag.
        
        :Args:
            * **plot_pred_int** (:py:class:`bool`): Flag to plot prediction interval
            * **prediction_intervals** (:py:class:`list`): Prediction intervals
            
        :Returns:
            * **prediction_intervals** (:py:class:`list` or `None`): Prediction intervals
        '''
        # check if prediction intervals exist and if user wants to plot them
        plot_pred_int = convert_flag_to_boolean(plot_pred_int)
        if plot_pred_int is False or prediction_intervals is None:
            prediction_intervals = None # turn off prediction intervals
        return prediction_intervals
    # --------------------------------------------
    @classmethod
    def _setup_labels(cls, prediction_intervals, nlines):
        '''
        Setup labels for prediction/credible intervals.
        
        :Args:
            * **prediction_intervals** (:py:class:`list`): Prediction intervals
            * **nlines** (:py:class:`int`): Number of lines
            
        :Returns:
            * **clabels** (:py:class:`list`): List of label strings for credible intervals.
            * **plabels** (:py:class:`list`): List of label strings for prediction intervals.
        '''
        clabels = ['95% CI']
        plabels = ['95% PI']
        
        # check if prediction intervals exist
        if prediction_intervals is None:
            clabels = ['99% CI', '95% CI', '90% CI', '50% CI']
            
        if prediction_intervals is None and nlines == 1:
            clabels = ['95% CI']
            
        return clabels, plabels
    # --------------------------------------------
    @classmethod
    def _setup_counting_metrics(cls, credible_intervals):
        '''
        Setup counting metrics for prediction/credible intervals.
        
        :Args:
            * **credible_intervals** (:py:class:`list`): Credible intervals
            
        :Returns:
            * **nbatch** (:py:class:`int`): Number of batches
            * **nn** (:py:class:`int`): Line number corresponding to median.
            * **nlines** (:py:class:`int`): Number of lines
        '''    
        # define number of batches
        nbatch = len(credible_intervals)
        
        # define counting metrics
        nlines = len(credible_intervals[0][0]) # number of lines
        nn = np.int((nlines + 1)/2) # median
        nlines = nn - 1
        
        return nbatch, nn, nlines
    # --------------------------------------------
    @classmethod
    def _observation_sample(cls, s2elem, ypred, sstype):
        '''
        Calculate model response with observation errors.
        
        :Args:
            * **s2elem** (:class:`~numpy.ndarray`): Observation error(s).
            * **ypred** (:class:`~numpy.ndarray`): Model responses.
            * **sstype** (:py:class:`int`): Flag to specify sstype.
            
        :Returns:
            * **opred** (:class:`~numpy.ndarray`): Model responses with observation errors.
        '''
        # check shape of s2elem and ypred
        ny = ypred.shape[1]
        ns = s2elem.shape[1]
        if ns != ny and ns == 1:
            s2elem = s2elem*np.ones([ny,1])
        elif ns != ny and ns != 1:
            sys.exit('Unclear data structure: error variances do not match size of model output')
            
        if sstype == 0:
            opred = ypred + np.matmul(np.random.standard_normal(ypred.shape),np.diagflat(
                    np.sqrt(s2elem))).reshape(ypred.shape)
        elif sstype == 1: # sqrt
            opred = (np.sqrt(ypred) + np.matmul(np.random.standard_normal(ypred.shape),np.diagflat(
                np.sqrt(s2elem))).reshape(ypred.shape))**2
        elif sstype == 2: # log
            opred = ypred*np.exp(np.matmul(np.random.standard_normal(ypred.shape),np.diagflat(
                np.sqrt(s2elem))).reshape(ypred.shape))
        else:
            sys.exit('Unknown sstype')
            
        return opred
    # --------------------------------------------
    @classmethod
    def _empirical_quantiles(cls, x, p = np.array([0.25, 0.5, 0.75])):
        '''
        Calculate empirical quantiles.
        
        :Args:
            * **x** (:class:`~numpy.ndarray`): Observations from which to generate quantile.
            * **p** (:class:`~numpy.ndarray`): Quantile limits.
            
        :Returns:
            * (:class:`~numpy.ndarray`): Interpolated quantiles.
        '''
    
        # extract number of rows/cols from np.array
        n = x.shape[0]
        # define vector valued interpolation function
        xpoints = range(n)
        interpfun = interp1d(xpoints, np.sort(x, 0), axis = 0)
        
        # evaluation points
        itpoints = (n-1)*p
        
        return interpfun(itpoints)