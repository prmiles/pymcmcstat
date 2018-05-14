#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:11:40 2018

@author: prmiles
"""
# import required packages
#import numpy as np
from .MetropolisAlgorithm import MetropolisAlgorithm
from .AdaptationAlgorithm import AdaptationAlgorithm
from .DelayedRejectionAlgorithm import DelayedRejectionAlgorithm

class SamplingMethods:
    def __init__(self):
        self.metropolis = MetropolisAlgorithm()
        self.delayed_rejection = DelayedRejectionAlgorithm()
        self.adaptation = AdaptationAlgorithm()