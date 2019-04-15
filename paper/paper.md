---
title: 'pymcmcstat: A Python package for Markov Chain Monte Carlo using Delayed Rejection Adaptive Metropolis'
tags:
  - Python
  - MCMC
  - DRAM
authors:
  - name: Paul R. Miles
    orcid: 0000-0002-7501-5114
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Mathematics, North Carolina State University
   index: 1
date: 15 April 2019
bibliography: paper.bib
---

# Summary
Bayesian parameter estimation is a powerful tool for quantifying model uncertainty, and Markov Chain Monte Carlo (MCMC) methods present a computationally tractable means of obtaining posterior distributions [@smith2014uncertainty].  A variety of Metropolis algorithms can be used within MCMC.  Ideally, as the sampling process progresses, information learned about the posterior distribution as candidate parameters are accepted will be used to update the proposal distribution.  This can be accomplished via a variety of adaptive Metropolis (AM) algorithms [@haario2001adaptive].  In addition to improving the proposal distribution via adaption, it is often advantageous to incorporate delayed rejection (DR) in order to stimulate mixing [@haario2006dram].

In the Python package ``pymcmcstat``, a variety of Metropolis based algorithms can be used for parameter estimation.  These algorithms include:

- Metropolis-Hastings (MH)
- Adaptive-Metropolis (AM): Adapts parameter covariance matrix at specified intervals.
- Delayed-Rejection (DR): Delays rejection by sampling from a narrower proposal distribution.
- Delayed Rejection Adaptive Metropolis (DRAM): DR + AM

The default program behavior is to always use 2-stage DRAM; however, it is capable of accomodating $n$-stage DR.  The specific algorithms implemented for AM and DR are as outlined in [@haario2001adaptive, @haario2006dram].

The ``pymcmcstat`` package was designed to be used by engineers and scientists interested in using Bayesian methods to quantify model parameter uncertainty.  Furthermore, the author desired to provide a comfortable Python platform for those researchers familiar with the MATLAB toolbox (mcmcstat)[https://mjlaine.github.io/mcmcstat/] developed by Marko Laine.  It is currently being used for several scientific projects, including radiation source localization using 3D transport models and fractional-order viscoelasticity models of dielectric elastomers.  The source code for ``pymcmcstat`` has been archived to Zenodo with the linked DOI: [@zenodo]

# Acknowledgements

This work was sponsored in part by the NNSA Office of Defense Nuclear Nonproliferation R&D through the Consortium for Nonproliferation Enabling Capabilities.

# References
