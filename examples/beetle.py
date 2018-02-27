#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:12:31 2017
% Example from Marko Laine's website: http://helios.fmi.fi/~lainema/mcmc/
%%
% <html><a href="../index.html">MCMC toolbox</a> » <a href="../examples.html">Examples</a> » Beetle</html>

%% Beetle mortality data
% From A. Dobson, _An Introduction to Generalized Linear Models_,
% Chapman & Hall/CRC, 2002.
% Binomial logistic regression example with dose-response data.
% See <beetless.html beetless.m> for -2log(likelihood) function.

%%
@author: prmiles
"""

# import required packages
from __future__ import division

#import math
import numpy as np
import math
from pymcmcstat.MCMC import MCMC
# for graphics
import matplotlib.pyplot as plt

# Beetle mortality data
dose = np.array([  1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839])
number_of_beetles = np.array([59, 60, 62, 56, 63, 59, 62, 60])
number_of_beetles_killed = np.array([6, 13, 18, 28, 52, 53, 61, 60])

x = np.array([dose, number_of_beetles])
y = number_of_beetles_killed

def logitmodelfun(d, t):
    return 1/(1+np.exp(t[0]+t[1]*d))
def loglogmodelfun(d, t):
    return 1 - np.exp(-np.exp(t[0] + t[1]*d))
def nordf(x, mu = 0, sigma2 = 1):
    # NORDF the standard normal (Gaussian) cumulative distribution.
    # NORPF(x,mu,sigma2) x quantile, mu mean, sigma2 variance
    # Marko Laine <Marko.Laine@Helsinki.FI>
    # $Revision: 1.5 $  $Date: 2007/12/04 08:57:00 $
    # adapted for Python by Paul Miles, November 8, 2017
    return 0.5 + 0.5*math.erf((x-mu)/math.sqrt(sigma2)/math.sqrt(2))
def probitmodelfun(d, t):
    tmp = np.vectorize(nordf)
    return tmp(t[0] + t[1]*d)

beetle_link_dictionary = {
        'logit': {'theta0': [60, -35], 'modelfun': logitmodelfun, 
                  'label': 'Beetle data with logit link'},
        'loglog': {'theta0': [-40, 22], 'modelfun': loglogmodelfun, 
                  'label': 'Beetle data with loglog link'},
        'probit': {'theta0': [-35, 20], 'modelfun': probitmodelfun, 
                  'label': 'Beetle data with loglog link'},
        }

# specify model type
beetle_link = 'loglog'        
beetle_model_object = beetle_link_dictionary[beetle_link]

# Initialize MCMC object
mcstat = MCMC()

# initialize data structure 
mcstat.data.add_data_set(x,y, user_defined_object=beetle_model_object)

# define sum-of-squares model function
def ssfun(theta,data):
    # unpack data
    ss = np.zeros([1])
    y = data.ydata[0]
    dose = data.xdata[0][0]
    n = data.xdata[0][1]
    obj = data.user_defined_object[0]
    model = obj['modelfun']
    
    # evaluate model
    p = model(dose, theta)
    
    # calculate loglikelihood
    ss = -2*sum(y*np.log(p) + (n-y)*np.log(1-p));
    
    return ss

# initialize parameter array
theta0 = beetle_model_object['theta0']
mcstat.parameters.add_model_parameter(name = '$b_0$', theta0 = theta0[0])
mcstat.parameters.add_model_parameter(name = '$b_1$', theta0 = theta0[1])

# Generate options
mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3))

# Define model object:
mcstat.model_settings.define_model_settings(sos_function = ssfun)

# Run mcmcrun
mcstat.run_simulation()

##
# Extract results
results = mcstat.simulation_results.results
names = results['names']
chain = results['chain']
s2chain = results['s2chain']
names = results['names'] # parameter names

# display chain stats
mcstat.chainstats(chain, results)

# plot chain panel
mcstat.mcmcplot.plot_chain_panel(chain, names)
plt.savefig('beetle_chain.eps', format = 'eps', dpi = 500)

# pairwise correlation
mcstat.mcmcplot.plot_pairwise_correlation_panel(chain, names)
plt.savefig('beetle_pairwise.eps', format = 'eps', dpi = 500)

# generate prediction and credible intervals
def predmodelfun(data, theta):
    dose = data.xdata[0]
    obj = data.user_defined_object[0]
    model = obj['modelfun']
    
    # evaluate model
    p = model(dose, theta)
    return p

# define data structure for prediction
predmcmc = MCMC()
xmod = np.linspace(1.5,2)
predmcmc.data.add_data_set(x = xmod, y = xmod, user_defined_object = beetle_model_object)

mcstat.PI.setup_prediction_interval_calculation(results = results, data = predmcmc.data, 
                                                modelfunction = predmodelfun)
mcstat.PI.generate_prediction_intervals(nsample = 500, calc_pred_int = 'off')
# plot credible intervals
plt.tight_layout()
mcstat.PI.plot_prediction_intervals(adddata = False)

plt.plot(dose, number_of_beetles_killed/number_of_beetles, 'ok', label = 'data'); # add data points to the plot
plt.xlabel('log(dose)',Fontsize=20)
plt.xticks(Fontsize=20)
plt.ylabel('proportion killed',Fontsize=20)
plt.yticks(Fontsize=20)
plt.title('Beetle data with loglog link',Fontsize=20)
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc='upper left')
plt.savefig('beetle_ci.eps', format = 'eps', dpi = 500, bbox_inches='tight')