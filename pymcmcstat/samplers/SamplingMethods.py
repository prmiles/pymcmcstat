#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:11:40 2018

@author: prmiles
"""
# import required packages
#import numpy as np
from .Metropolis import Metropolis
from .Adaptation import Adaptation
from .DelayedRejection import DelayedRejection

class SamplingMethods:
    '''
    Metropolis sampling methods.
    
    Attributes:
        * :class:`~.Metropolis`
        * :class:`~.DelayedRejection`
        * :class:`~.Adaptation`
    '''
    def __init__(self):
        self.metropolis = Metropolis()
        self.delayed_rejection = DelayedRejection()
        self.adaptation = Adaptation()