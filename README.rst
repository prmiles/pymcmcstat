`pymcmcstat`
============

|docs| |build| |coverage| |license| |codacy| |zenodo|

The `pymcmcstat` package is a Python program for running Markov Chain Monte Carlo (MCMC) simulations.
Included in this package is the ability to use different Metropolis based sampling techniques:

* Metropolis-Hastings (MH): Primary sampling method.
* Adaptive-Metropolis (AM): Adapts covariance matrix at specified intervals.
* Delayed-Rejection (DR): Delays rejection by sampling from a narrower distribution.  Capable of `n`-stage delayed rejection.
* Delayed Rejection Adaptive Metropolis (DRAM): DR + AM

The `pymcmcstat homepage <https://prmiles.wordpress.ncsu.edu/codes/python-packages/pymcmcstat/>`_ contains tutorials for users as well as installation instructions.

Python implementation of MATLAB toolbox "mcmcstat".  This code is designed to replicate the functionality of the MATLAB routines developed and posted here: http://helios.fmi.fi/~lainema/mcmc/

The user interface is designed to be as similar to the MATLAB version as possible, but this implementation has taken advantage of certain data structure concepts more amenable to Python.  

Installation
============

This code can be found on the `Github project page <https://github.com/prmiles/pymcmcstat>`_.  It is open source and provided under the MIT license.
To install directly from Github,

::

    pip install git+https://github.com/prmiles/pymcmcstat.git

You can also clone the repository and run ``python  setup.py install``.

Package is also available on the PyPI distribution site.

::

    pip install pymcmcstat

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
   
.. |docs| image:: https://readthedocs.org/projects/pymcmcstat/badge/?version=latest
    :target: https://pymcmcstat.readthedocs.io/en/latest/?badge=latest
    :scale: 100%
    
.. |build| image:: https://travis-ci.org/prmiles/pymcmcstat.svg?branch=master
    :target: https://travis-ci.org/prmiles/pymcmcstat
    :scale: 100%
    
.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://github.com/prmiles/pymcmcstat/blob/master/LICENSE.txt

.. |coverage| image:: https://coveralls.io/repos/github/prmiles/pymcmcstat/badge.svg
    :target: https://coveralls.io/github/prmiles/pymcmcstat

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/b1a33340c57a47648f993e124c75e93a    
    :target: https://www.codacy.com/app/prmiles/pymcmcstat?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=prmiles/pymcmcstat&amp;utm_campaign=Badge_Grade

.. |zenodo| image:: https://zenodo.org/badge/107596954.svg
    :target: https://zenodo.org/badge/latestdoi/107596954