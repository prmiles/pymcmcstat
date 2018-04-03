#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:11:40 2018

@author: prmiles
"""
# import required packages
#import numpy as np
from pymcmcstat import MetropolisAlgorithm, AdaptationAlgorithm, DelayedRejectionAlgorithm

class SamplingMethods:
    def __init__(self):
        self.metropolis = MetropolisAlgorithm.MetropolisAlgorithm()
        self.delayed_rejection = DelayedRejectionAlgorithm.DelayedRejectionAlgorithm()
        self.adaptation = AdaptationAlgorithm.AdaptationAlgorithm()