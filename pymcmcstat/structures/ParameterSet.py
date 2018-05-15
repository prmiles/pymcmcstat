#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:15:37 2018

Description: Storage device for passing parameter sets back and forth between sampling methods

@author: prmiles
"""

class ParameterSet:
    def __init__(self, theta = None, ss= None, prior = None, sigma2 = None, alpha = None):
        self.theta = theta
        self.ss = ss
        self.prior = prior
        self.sigma2 = sigma2
        self.alpha = alpha