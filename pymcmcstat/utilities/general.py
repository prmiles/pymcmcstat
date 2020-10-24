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


# --------------------------
def format_number_to_str(number):
    '''
    Format number for display

    Args:
        * **number** (:py:class:`float`): Number to be formatted

    Returns:
        * (:py:class:`str`): Formatted string display
    '''
    if abs(number) >= 1e4 or abs(number) <= 1e-2:
        return str('{:9.2e}'.format(number))
    else:
        return str('{:9.2f}'.format(number))


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


def check_settings(default_settings, user_settings=None):
    '''
    Check user settings with default.

    Recursively checks elements of user settings against the defaults and updates settings
    as it goes.  If a user setting does not exist in the default, then the user setting
    is added to the settings.  If the setting is defined in both the user and default
    settings, then the user setting overrides the default.  Otherwise, the default
    settings persist.

    Args:
        * **default_settings** (:py:class:`dict`): Default settings for particular method.
        * **user_settings** (:py:class:`dict`): User defined settings.

    Returns:
        * (:py:class:`dict`): Updated settings.
    '''
    settings = default_settings.copy()  # initially define settings as default
    options = list(default_settings.keys())  # get default settings
    if user_settings is None:  # convert to empty dict
        user_settings = {}
    user_options = list(user_settings.keys())  # get user settings
    for uo in user_options:  # iterate through settings
        if uo in options:
            # check if checking a dictionary
            if isinstance(settings[uo], dict):
                settings[uo] = check_settings(settings[uo], user_settings[uo])
            else:
                settings[uo] = user_settings[uo]
        if uo not in options:
            settings[uo] = user_settings[uo]
    return settings
