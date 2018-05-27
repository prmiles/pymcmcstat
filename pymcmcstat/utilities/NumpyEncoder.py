#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 08:42:56 2018

@author: prmiles
"""

import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    '''
    Encoder used for storing numpy arrays in json files.
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)