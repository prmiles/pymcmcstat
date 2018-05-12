.. pymcmcstat documentation master file, created by
   sphinx-quickstart on Thu May 10 06:36:58 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pymcmcstat's documentation!
========================================
The `pymcmcstat` package is a Python program for running Markov Chain Monte Carlo (MCMC) simulations.
Included in this package is the abilitity to use different Metropolis based sampling techniques:

* Metropolis-Hastings (MH): Primary sampling method.
* Adaptive-Metropolis (AM): Adapts covariance matrix at specified intervals.
* Delayed-Rejection (DR): Delays rejection by sampling froma  narrower distribution.  Capable of :math:`n`-stage delayed rejection.
* Delayed Rejection Adaptive Metropolis (DRAM): DR + AM

The `pymcmcstat homepage <https://prmiles.wordpress.ncsu.edu/codes/python-packages/pymcmcstat/>`_ contains tutorials for users as well as installation instructions.

This code can be found on the `Github project page <https://github.com/prmiles/pymcmcstat>`_.  It is open sources and provided under the MIT license.

Contents:

.. toctree::
   :maxdepth: 2
   
   pymcmcstat

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
