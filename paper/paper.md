---
title: 'pymcmcstat: A Python Package for Bayesian Inference Using Delayed Rejection Adaptive Metropolis'
tags:
  - Python
  - MCMC
  - DRAM
  - Bayesian Inference
authors:
  - name: Paul R. Miles
    orcid: 0000-0002-7501-5114
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Mathematics, North Carolina State University
   index: 1
date: 22 April 2019
bibliography: paper.bib
---

# Summary
Bayesian inference is a powerful tool for quantifying model input uncertainty, and Markov Chain Monte Carlo (MCMC) methods present a computationally tractable means for constructing posterior distributions for input parameters [@smith2014uncertainty].  A variety of Metropolis algorithms can be used within MCMC.  Ideally, information learned about the posterior distribution as candidate parameters are accepted will be used to update the proposal distribution.  This can be accomplished via a variety of adaptive Metropolis (AM) algorithms [@andrieu2008tutorial, @haario2001adaptive, @roberts2009examples].  In addition to improving the proposal distribution via adaption, it is often advantageous to incorporate delayed rejection (DR) in order to stimulate mixing [@haario2006dram].  Both mechanisms have been demonstrated to significantly improve the performance of MCMC simulations.

In the Python package ``pymcmcstat``, we employ various Metropolis-based algorithms for parameter estimation.  These algorithms include:

- Metropolis-Hastings (MH)
- Adaptive-Metropolis (AM): Adapts parameter covariance matrix at specified intervals.
- Delayed-Rejection (DR): Delays rejection by sampling from a narrower proposal distribution.
- Delayed Rejection Adaptive Metropolis (DRAM): DR + AM

The default program employs 2-stage DRAM; however, it is capable of accomodating $n$-stage DR.  The specific algorithms implemented for AM and DR are outlined in [@haario2001adaptive, @haario2006dram].

The ``pymcmcstat`` package was designed for engineers and scientists interested in using Bayesian methods to quantify model parameter uncertainty.  Furthermore, we had the goal of providing a Python platform for researchers familiar with the MATLAB toolbox [mcmcstat](https://mjlaine.github.io/mcmcstat/) developed by Marko Laine.  To accommodate a diverse audience, we constructed several tutorials to guide the user through the various stages of setting up a problem, such as defining the data structure, model parameters, and simulation options.  Currently, the package is limited to Gaussian likelihood and prior functions; however, these are still suitable for a wide variety of scientific problems.  

To the author's knowledge, the package is being used for several scientific projects, including radiation source localization using 3D transport models and fractional-order viscoelasticity models of dielectric elastomers.  The source code for ``pymcmcstat`` has been archived to Zenodo with the linked DOI: [@zenodo]

# Acknowledgements

This work was sponsored in part by the NNSA Office of Defense Nuclear Nonproliferation R&D through the Consortium for Nonproliferation Enabling Capabilities.

# References
