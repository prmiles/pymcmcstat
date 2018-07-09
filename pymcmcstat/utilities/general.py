#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 06:16:02 2018

General functions used throughout package

@author: prmiles
"""

def message(verbosity, level, printthis):
    '''
    Display message

    :Args:
        * **verbosity** (:py:class:`int`): Verbosity of display output.
        * **level** (:py:class:`int`): Print level relative to verbosity.
        * **printthis** (:py:class:`str`): String to be printed.
    '''
    printed = False
    if verbosity >= level:
        print(printthis)
        printed = True
    return printed