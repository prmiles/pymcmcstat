`pymcmcstat`
============

|docs| |build| |coverage| |license|

The `pymcmcstat` package is a Python program for running Markov Chain Monte Carlo (MCMC) simulations.
Included in this package is the ability to use different Metropolis based sampling techniques:

* Metropolis-Hastings (MH): Primary sampling method.
* Adaptive-Metropolis (AM): Adapts covariance matrix at specified intervals.
* Delayed-Rejection (DR): Delays rejection by sampling from a narrower distribution.  Capable of :math:`n`-stage delayed rejection.
* Delayed Rejection Adaptive Metropolis (DRAM): DR + AM

The `pymcmcstat homepage <https://prmiles.wordpress.ncsu.edu/codes/python-packages/pymcmcstat/>`_ contains tutorials for users as well as installation instructions.

Python implementation of Matlab package "mcmcstat".  This code is designed to replicate the functionality of the Matlab routines developed and posted here: http://helios.fmi.fi/~lainema/mcmc/

The user interface is designed to be as similar to the Matlab version as possible, but this implementation has taken advantage of certain data structure concepts more amenable to python.  

Installation
============

This code can be found on the `Github project page <https://github.com/prmiles/pymcmcstat>`_.  It is open sources and provided under the MIT license.
To install directly from Github,
::
    pip install git+https://github.com/prmiles/pymcmcstat.git

You can also clone the repository and run ``python  setup.py install``.

Package is currently in the process of being added to the PyPI distribution site.

Getting Started
===============

- Tutorial `notebooks <https://nbviewer.jupyter.org/github/prmiles/notebooks/tree/master/pymcmcstat/index.ipynb>`_
- `Documentation <http://pymcmcstat.readthedocs.io/>`_

License
=======

`MIT <https://github.com/prmiles/pymcmcstat/blob/master/LICENSE.txt>`_

Contributors
============

See the `GitHub contributor
page <https://github.com/prmiles/pymcmcstat/graphs/contributors>`_
   
.. |docs| image:: https://readthedocs.org/projects/pymcmcstat/badge/?version=master
    :target: https://pymcmcstat.readthedocs.io/en/latest/?badge=latest
    :scale: 100%
    
.. |build| image:: https://travis-ci.org/prmiles/pymcmcstat.svg?branch=master
    :target: https://travis-ci.org/prmiles/pymcmcstat
    :scale: 100%
    
.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://github.com/prmiles/pymcmcstat/blob/master/LICENSE.txt

.. |coverage| image:: https://coveralls.io/repos/github/prmiles/pymcmcstat/badge.svg?branch=master
    :target: https://coveralls.io/github/prmiles/pymcmcstat?branch=master


