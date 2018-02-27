#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:12:31 2017
% Example from Marko Laine's website: http://helios.fmi.fi/~lainema/mcmc/
%% MCMC toolbox examples
% This example is from 
% P. M. Berthouex and L. C. Brown:
% _Statistics for Environmental Engineers_, CRC Press, 2002.
%
% We fit the Monod model
%
% $$ y = \theta_1 \frac{t}{\theta_2 + t} + \epsilon \quad \epsilon\sim N(0,I\sigma^2) $$
%
% to observations
%
%   x (mg / L COD):  28    55    83    110   138   225   375   
%   y (1 / h):       0.053 0.060 0.112 0.105 0.099 0.122 0.125
%

%%
@author: prmiles
"""

# import required packages
from __future__ import division
import numpy as np
import scipy.optimize
from pymcmcstat.MCMC import MCMC
# for graphics
import matplotlib.pyplot as plt

# Initialize MCMC object
mcstat = MCMC()

# Next, create a data structure for the observations and control
# variables. Typically one could make a structure |data| that
# contains fields |xdata| and |ydata|.
ndp = 7
x = np.array([28,    55,    83,    110,   138,   225,   375])   # (mg / L COD)
x = x.reshape(ndp,1) # enforce column vector
y = np.array([0.053, 0.060, 0.112, 0.105, 0.099, 0.122, 0.125]) # (1 / h)
y = y.reshape(ndp,1) # enforce column vector

# data structure 
mcstat.data.add_data_set(x,y)

# Here is a plot of the data set.
plt.figure(1)
hdata, = plt.plot(mcstat.data.xdata[0], mcstat.data.ydata[0], 's', label = 'data');
plt.xlim([0, 400])
plt.xlabel('x (mg/L COD)')
plt.ylabel('y (1/h)')
plt.savefig('monod_data.eps', format = 'eps', dpi = 500)

# For the MCMC run we need the sum of squares function. For the
# plots we shall also need a function that returns the model.
# Both the model and the sum of squares functions are
# easy to write as follows

def modelfun(x, theta):
    return theta[0]*x/(theta[1] + x)

def ssfun(theta,data):
    return sum((data.ydata[0] - modelfun(data.xdata[0], theta))**2)

# In this case the initial values for the parameters are easy to guess
# by looking at the plotted data. We can easily define a residuals function and
# we might as well try to minimize it using scipy's optimize least-squares.
def residuals(p, x, y):
    return y - modelfun(x, p)

theta0, ssmin = scipy.optimize.leastsq(residuals, x0 = [0.15, 100], args=(
        mcstat.data.xdata[0].reshape(ndp,), mcstat.data.ydata[0].reshape(ndp,)))
n = mcstat.data.n[0] # number of data points in model
p = len(theta0); # number of model parameters (dof)
ssmin = ssfun(theta0, mcstat.data) # calculate the sum-of-squares error
mse = ssmin/(n-p) # estimate for the error variance

# The Jacobian matrix of the model function is easy to calculate so we use
# it to produce estimate of the covariance of theta. This can be
# used as the initial proposal covariance for the MCMC samples by
# option |options.qcov| below.
J = np.array([[x/(theta0[1]+x)], [-theta0[0]*x/(theta0[1]+x)**2]])
J = J.transpose()
J = J.reshape(n,p)
#tcov = inv(J'*J)*mse
tcov = np.linalg.inv(np.dot(J.transpose(),J))*mse
print('tcov = {}'.format(tcov))

# We have to define three structures for inputs of the |mcmcrun|
# function: parameter, model, and options.  Parameter structure has a
# special form and it is constructed using the Parameters class. 
# At least, the structure has the name and the initial value for each parameter. 
# Third optional parameter given below is the minimal accepted value. 
# With it we set a positivity constraits for both of the parameters.
## add model parameters
mcstat.parameters.add_model_parameter(name = '$\mu_{max}$', theta0 = theta0[0], minimum = 0)
mcstat.parameters.add_model_parameter(name = '$K_x$', theta0 = theta0[1], minimum = 0)

# The |options| structure has settings for the MCMC run. We need at
# least the number of simulations in |nsimu|. Here we also set the
# option |updatesigma| to allow automatic sampling and estimation of the
# error variance. The option |qcov| sets the initial covariance for
# the Gaussian proposal density of the MCMC sampler.
# Generate options
mcstat.simulation_options.define_simulation_options(
        nsimu = int(5.0e3), updatesigma = 1, qcov = tcov)

# The |model| structure holds information about the model. Minimally
# we need to set |ssfun| for the sum of squares function and the
# initial estimate of the error variance |sigma2|.
# Define model object:
mcstat.model_settings.define_model_settings(sos_function = ssfun, sigma2 = 0.01**2)

# The actual MCMC simulation run is done using the function
# |mcmcrun|.
# Run mcmcrun
mcstat.run_simulation()

##
# After the run the we have a dictionary structure |results| that contains some
# information about the run, including matrix outputs |chain| and
# |s2chain| that contain the actual MCMC chains for the parameters
# and for the observation error variance.
# Extract results
results = mcstat.simulation_results.results
names = results['names']

##
# The |chain| variable is a |nsimu| × |npar| matrix and it can be
# plotted and manipulated with standard plotting functions. The MCMC package
# function |mcmcplot| can be used to make some useful chain plots and
# also plot 1 dimensional marginal kernel density estimates of
# the posterior distributions.

chain = results['chain']
s2chain = results['s2chain']
names = results['names'] # parameter names

# plot chain panel
mcstat.mcmcplot.plot_chain_panel(chain, names)
plt.savefig('monod_chainpanel.eps', format = 'eps', dpi = 500)
# The |'pairs'| options makes pairwise scatterplots of the columns of
# the |chain|.
mcstat.mcmcplot.plot_pairwise_correlation_panel(chain, names)
plt.savefig('monod_pairwise.eps', format = 'eps', dpi = 500)
# If we take square root of the |s2chain| we get the chain for error
# standard deviation. Here we use |'hist'| option for the histogram of
# the chain.
mcstat.mcmcplot.plot_histogram_panel(chains = np.sqrt(s2chain), names = 'sigma_2')
plt.title('Error std posterior')
plt.savefig('monod_std_hist.eps', format = 'eps', dpi = 500)
# A point estimate of the model parameters can be calculated from the
# mean of the |chain|. Here we plot the fitted model using the
# posterior means of the parameters.
xmod = np.linspace(1e-4,400)
plt.figure(1)
hmodel, = plt.plot(xmod,modelfun(xmod,np.mean(chain, 0)),'-k', label = 'model')
plt.legend(handles = [hdata, hmodel])
plt.savefig('monod_data_model.eps', format = 'eps', dpi = 500)

# Instead of just a point estimate of the fit, we should also study
# the predictive posterior distribution of the model. The |mcmcpred|
# and |mcmcpredplot| functions can be used for this purpose. By them
# we can calculate the model fit for a randomly selected subset of the
# chain and calculate the predictive envelope of the model. The grey
# areas in the plot correspond to 50%, 90%, 95%, and 99% posterior
# regions.

def predmodelfun(data, theta):
    return modelfun(data.xdata[0], theta)

mcstat.PI.setup_prediction_interval_calculation(results = results, data = mcstat.data, 
                                                modelfunction = predmodelfun)
mcstat.PI.generate_prediction_intervals(nsample = 500, calc_pred_int = 'off')
# plot prediction intervals
mcstat.PI.plot_prediction_intervals(adddata = True)
plt.xlabel('x (mg/L COD)',Fontsize=20)
plt.xticks(Fontsize=20)
plt.ylabel('y (1/h)',Fontsize=20)
plt.yticks(Fontsize=20)
plt.title('Predictive envelopes of the model',Fontsize=20)
plt.savefig('monod_ci.eps', format = 'eps', dpi = 500, bbox_inches='tight')