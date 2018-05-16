# pymcmcstat

The `pymcmcstat` package is a Python program for running Markov Chain Monte Carlo (MCMC) simulations.
Included in this package is the abilitity to use different Metropolis based sampling techniques:

* Metropolis-Hastings (MH): Primary sampling method.
* Adaptive-Metropolis (AM): Adapts covariance matrix at specified intervals.
* Delayed-Rejection (DR): Delays rejection by sampling froma  narrower distribution.  Capable of :math:`n`-stage delayed rejection.
* Delayed Rejection Adaptive Metropolis (DRAM): DR + AM

The `pymcmcstat homepage <https://prmiles.wordpress.ncsu.edu/codes/python-packages/pymcmcstat/>`_ contains tutorials for users as well as installation instructions.

This code can be found on the `Github project page <https://github.com/prmiles/pymcmcstat>`_.  It is open sources and provided under the MIT license.
Python implementation of Matlab package "mcmcstat"

This code is designed to replicate the functionality of the Matlab routines developed and posted here: http://helios.fmi.fi/~lainema/mcmc/

The user interface is designed to be as similar to the Matlab version as possible, but this implementation has taken advantage of certain data structure concepts more amenable to python.  Documentation for the package can be found here: https://prmiles.wordpress.ncsu.edu/codes/

Example scripts for using this package can be found here: https://github.com/prmiles/pymcmcstat_example_scripts
