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

    Args:
        * **verbosity** (:py:class:`int`): Verbosity of display output.
        * **level** (:py:class:`int`): Print level relative to verbosity.
        * **printthis** (:py:class:`str`): String to be printed.
    '''
    printed = False
    if verbosity >= level:
        print(printthis)
        printed = True
    return printed


def removekey(d, key):
    '''
    Removed elements from dictionary and return the remainder.

    Args:
        * **d** (:py:class:`dict`): Original dictionary.
        * **key** (:py:class:`str`): Keyword to be removed.

    Returns:
        * **r** (:py:class:`dict`): Updated dictionary without the keyword, value pair.
    '''
    r = dict(d)
    del r[key]
    return r
