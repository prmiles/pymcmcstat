#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:00:11 2017

@author: prmiles
"""

import numpy as np
import sys
from .utilities.progressbar import progress_bar
from .plotting.utilities import check_settings
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mplcolor
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d


def check_s2chain(s2chain, nsimu):
    '''
    Check size of s2chain
    
    Args:
        * **s2chain** (:py:class:`float`, :class:`~numpy.ndarray`, or `None`):
            Observation error variance chain or value
        * **nsimu** (:py:class:`int`): No. of elements in chain

    Returns:
        * **s2chain** (:class:`~numpy.ndarray` or `None`)

    Raises:
        * System exit if it is an array that size is > nsimu.
    '''
    if s2chain is None:
        return None
    else:
        if isinstance(s2chain, float):
            s2chain = np.ones((nsimu,))*s2chain

        if s2chain.size == nsimu:
            return s2chain
        else:
            sys.exit('Expect s2chain as float or array of size nsimu')


# --------------------------------------------
def observation_sample(s2, y, sstype):
    '''
    Calculate model response with observation errors.

    Args:
        * **s2** (:class:`~numpy.ndarray`): Observation error(s).
        * **y** (:class:`~numpy.ndarray`): Model responses.
        * **sstype** (:py:class:`int`): Flag to specify sstype.

    Returns:
        * **opred** (:class:`~numpy.ndarray`): Model responses with observation errors.
    '''
    if sstype == 0:
        opred = y + np.random.standard_normal(y.shape) * np.sqrt(s2)
    elif sstype == 1:  # sqrt
        opred = (np.sqrt(y) + np.random.standard_normal(y.shape) * np.sqrt(s2))**2
    elif sstype == 2:  # log
        opred = y*np.exp(np.random.standard_normal(y.shape) * np.sqrt(s2))
    else:
        sys.exit('Unknown sstype')
    return opred


# --------------------------------------------
def define_sample_points(nsample, nsimu):
    '''
    Define indices to sample from posteriors.

    Args:
        * **nsample** (:py:class:`int`): Number of samples to draw from posterior.
        * **nsimu** (:py:class:`int`): Number of MCMC simulations.

    Returns:
        * **iisample** (:class:`~numpy.ndarray`): Array of indices in posterior set.
        * **nsample** (:py:class:`int`): Number of samples to draw from posterior.
    '''
    # define sample points
    if nsample >= nsimu:
        iisample = range(nsimu)  # sample all points from chain
        nsample = nsimu
    else:
        # randomly sample from chain
        iisample = np.ceil(np.random.rand(nsample)*nsimu) - 1
        iisample = iisample.astype(int)
    return iisample, nsample


# --------------------------------------------
def generate_quantiles(x, p=np.array([0.25, 0.5, 0.75])):
    '''
    Calculate empirical quantiles.

    Args:
        * **x** (:class:`~numpy.ndarray`): Observations from which to generate quantile.
        * **p** (:class:`~numpy.ndarray`): Quantile limits.

    Returns:
        * (:class:`~numpy.ndarray`): Interpolated quantiles.
    '''
    # extract number of rows/cols from np.array
    n = x.shape[0]
    # define vector valued interpolation function
    xpoints = np.arange(0, n, 1)
    interpfun = interp1d(xpoints, np.sort(x, 0), axis=0)
    # evaluation points
    itpoints = (n - 1)*p
    return interpfun(itpoints)


def setup_display_settings(interval_display, model_display, data_display):
    '''
    Compare user defined display settings with defaults and merge.

    Args:
        * **interval_display** (:py:class:`dict`): User defined settings for interval display.
        * **model_display** (:py:class:`dict`): User defined settings for model display.
        * **data_display** (:py:class:`dict`): User defined settings for data display.

    Returns:
        * **interval_display** (:py:class:`dict`): Settings for interval display.
        * **model_display** (:py:class:`dict`): Settings for model display.
        * **data_display** (:py:class:`dict`): Settings for data display.
    '''
    # Setup interval display
    default_interval_display = dict(
            linestyle=':',
            linewidth=1,
            alpha=1.0,
            edgecolor='k')
    interval_display = check_settings(default_interval_display, interval_display)
    # Setup model display
    default_model_display = dict(
            linestyle='-',
            color='r',
            marker='',
            linewidth=2,
            markersize=5,
            label='Model')
    model_display = check_settings(default_model_display, model_display)
    # Setup data display
    default_data_display = dict(
            linestyle='',
            color='b',
            marker='.',
            linewidth=1,
            markersize=5,
            label='Data')
    data_display = check_settings(default_data_display, data_display)
    return interval_display, model_display, data_display


def setup_interval_colors(iset, inttype='CI'):
    '''
    Setup colors for empirical intervals
    
    This routine attempts to distribute the color of the UQ intervals
    based on a normalize color map.  Or, it will assign user-defined
    colors; however, this only happens if the correct number of colors
    are specified.

    Args:
        * **iset** (:py:class:`dict`):  This dictionary should contain the
          following keys - `limits`, `cmap`, and `colors`.

    Kwargs:
        * **inttype** (:py:class:`str`): Type of uncertainty interval

    Returns:
        * **ic** (:py:class:`list`): List containing color for each interval
    '''
    limits, cmap, colors = iset['limits'], iset['cmap'], iset['colors']
    norm = __setup_cmap_norm(limits)
    cmap = __setup_default_cmap(cmap, inttype)
    # assign colors using color map or using colors defined by user
    ic = []
    if colors is None:  # No user defined colors
        for limits in limits:
            ic.append(cmap(norm(limits)))
    else:
        if len(colors) == len(limits):  # correct number of colors defined
            for color in colors:
                ic.append(color)
        else:  # User defined the wrong number of colors
            print('Note, user-defined colors were ignored. Using color map. '
                  + 'Expected a list of length {}, but received {}'.format(
                          len(limits), len(colors)))
            for limits in limits:
                ic.append(cmap(norm(limits)))
    return ic


# --------------------------------------------
def _setup_labels(limits, inttype='CI'):
    '''
    Setup labels for prediction/credible intervals.
    '''
    labels = []
    for limit in limits:
        labels.append(str('{}% {}'.format(limit, inttype)))
    return labels


def _check_limits(limits, default_limits):
    if limits is None:
        limits = default_limits
    limits.sort(reverse=True)
    return limits


def _convert_limits(limits):
    rng = []
    for limit in limits:
        limit = limit/100
        rng.append([0.5 - limit/2, 0.5 + limit/2])
    return rng


def __setup_cmap_norm(limits):
    if len(limits) == 1:
        norm = mplcolor.Normalize(vmin=0, vmax=100)
    else:
        norm = mplcolor.Normalize(vmin=min(limits), vmax=max(limits))
    return norm


def __setup_default_cmap(cmap, inttype):
    if cmap is None:
        if inttype.upper() == 'CI':
            cmap = cm.autumn
        else:
            cmap = cm.winter
    return cmap


# ******************************************************
def calculate_intervals(chain, results, data, model, s2chain=None,
                        nsample=500, waitbar=True, sstype=0):
    parind = results['parind']
    q = results['theta']
    nsimu, npar = chain.shape
    s2chain = check_s2chain(s2chain, nsimu)
    iisample, nsample = define_sample_points(nsample, nsimu)
    if waitbar is True:
        __wbarstatus = progress_bar(iters=int(nsample))

    ci = []
    pi = []
    for kk, isa in enumerate(iisample):
        # progress bar
        if waitbar is True:
            __wbarstatus.update(kk)
        # extract chain set
        q[parind] = chain[kk, :]
        # evaluate model
        y = model(q, data)
        # store model prediction in credible intervals
        ci.append(y.reshape(y.size,))  # store model output
        if s2chain is None:
            continue
        else:
            # estimate prediction intervals
            s2 = s2chain[kk]
            obs = observation_sample(s2, y, sstype)
            pi.append(obs.reshape(obs.size,))

    # Setup output
    credible = np.array(ci)
    if s2chain is None:
        prediction = None
    else:
        prediction = np.array(pi)
    return dict(credible=credible,
                prediction=prediction)


# --------------------------------------------
def plot_intervals(intervals, time, ydata, limits=[50, 90, 95, 99],
                   adddata=False, addlegend=True,
                   addmodel=True, figsize=None, model_display={},
                   data_display={}, interval_display={},
                   addcredible=True, addprediction=True,
                   fig=None, legloc='upper left',
                   ciset=dict(limits=None, cmap=None, colors=None),
                   piset=dict(limits=None, cmap=None, colors=None),
                   return_settings=False):
    '''
    Plot propagation intervals
    '''
    # unpack dictionary
    credible = intervals['credible']
    prediction = intervals['prediction']
    # Check user-defined settings
    if ciset['limits'] is None:
        ciset['limits'] = limits
    if piset['limits'] is None:
        piset['limits'] = limits
    # convert limits to ranges
    ciset['quantiles'] = _convert_limits(ciset['limits'])
    piset['quantiles'] = _convert_limits(piset['limits'])
    # setup display settings
    interval_display, model_display, data_display = setup_display_settings(
            interval_display, model_display, data_display)
    # Define colors
    ciset['colors'] = setup_interval_colors(ciset, inttype='ci')
    piset['colors'] = setup_interval_colors(piset, inttype='pi')
    # Define labels
    ciset['labels'] = _setup_labels(ciset['limits'], inttype='CI')
    piset['labels'] = _setup_labels(piset['limits'], inttype='PI')
    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    time = time.reshape(time.size,)
    # add prediction intervals
    if addprediction is True:
        for ii, quantile in enumerate(piset['quantiles']):
            pi = generate_quantiles(prediction, np.array(quantile))
            ax.fill_between(time, pi[0], pi[1], facecolor=piset['colors'][ii],
                            label=piset['labels'][ii], **interval_display)
    # add credible intervals
    if addcredible is True:
        for ii, quantile in enumerate(ciset['quantiles']):
            ci = generate_quantiles(credible, np.array(quantile))
            ax.fill_between(time, ci[0], ci[1], facecolor=ciset['colors'][ii],
                            label=ciset['labels'][ii], **interval_display)
    # add model (median model response)
    if addmodel is True:
        ci = generate_quantiles(credible, np.array(0.5))
        ax.plot(time, ci, **model_display)
    # add data to plot
    if adddata is True:
        plt.plot(time, ydata, **data_display)
    # add legend
    if addlegend is True:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=legloc)
    return fig, ax


# --------------------------------------------
def plot_3d_intervals(intervals, time, ydata, limits=[50, 90, 95, 99],
                      adddata=False, addlegend=True,
                      addmodel=True, figsize=None, model_display={},
                      data_display={}, interval_display={},
                      addcredible=True, addprediction=True,
                      fig=None, legloc='upper left',
                      ciset=dict(limits=None, cmap=None, colors=None),
                      piset=dict(limits=None, cmap=None, colors=None),
                      return_settings=False):
    '''
    Plot propagation intervals
    '''
    # unpack dictionary
    credible = intervals['credible']
    prediction = intervals['prediction']
    # Check user-defined settings
    if ciset['limits'] is None:
        ciset['limits'] = limits
    if piset['limits'] is None:
        piset['limits'] = limits
    # convert limits to ranges
    ciset['quantiles'] = _convert_limits(ciset['limits'])
    piset['quantiles'] = _convert_limits(piset['limits'])
    # setup display settings
    interval_display, model_display, data_display = setup_display_settings(
            interval_display, model_display, data_display)
    # Define colors
    ciset['colors'] = setup_interval_colors(ciset, inttype='ci')
    piset['colors'] = setup_interval_colors(piset, inttype='pi')
    # Define labels
    ciset['labels'] = _setup_labels(ciset['limits'], inttype='CI')
    piset['labels'] = _setup_labels(piset['limits'], inttype='PI')
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
    ax = fig.gca()
    time1 = time[:, 0]
    time2 = time[:, 1]
    # add prediction intervals
    if addprediction is True:
        for ii, quantile in enumerate(piset['quantiles']):
            pi = generate_quantiles(prediction, np.array(quantile))
            # Add a polygon instead of fill_between
            rev = np.arange(time1.size - 1, -1, -1)
            x = np.concatenate((time1, time1[rev]))
            y = np.concatenate((time2, time2[rev]))
            z = np.concatenate((pi[0], pi[1][rev]))
            verts = [list(zip(x, y, z))]
            surf = Poly3DCollection(verts,
                                    color=piset['colors'][ii],
                                    label=piset['labels'][ii])
            # Add fix for legend compatibility
            surf._facecolors2d = surf._facecolors3d
            surf._edgecolors2d = surf._edgecolors3d
            ax.add_collection3d(surf)
    # add credible intervals
    if addcredible is True:
        for ii, quantile in enumerate(ciset['quantiles']):
            ci = generate_quantiles(credible, np.array(quantile))
            # Add a polygon instead of fill_between
            rev = np.arange(time1.size - 1, -1, -1)
            x = np.concatenate((time1, time1[rev]))
            y = np.concatenate((time2, time2[rev]))
            z = np.concatenate((ci[0], ci[1][rev]))
            verts = [list(zip(x, y, z))]
            surf = Poly3DCollection(verts,
                                    color=ciset['colors'][ii],
                                    label=ciset['labels'][ii])
            # Add fix for legend compatibility
            surf._facecolors2d = surf._facecolors3d
            surf._edgecolors2d = surf._edgecolors3d
            ax.add_collection3d(surf)
    # add model (median model response)
    if addmodel is True:
        ci = generate_quantiles(credible, np.array(0.5))
        ax.plot(time1, time2, ci, **model_display)
    # add data to plot
    if adddata is True:
        ax.plot(time1, time2, ydata.reshape(time1.shape), **data_display)
    # add legend
    if addlegend is True:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=legloc)
    if return_settings is True:
        return fig, ax, dict(ciset=ciset, piset=piset)
    else:
        return fig, ax
