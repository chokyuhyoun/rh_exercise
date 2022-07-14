#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:39:21 2021

@author: chokh2
"""

import numpy as np

def f(x, S):
    return x-np.sin(S*x + theta1)

S = 0.5
phi = 0
P = 2.7
omega = 2*np.pi/(P*60)
t = 0
theta = omega*t - phi
theta1 = theta // 2*np.pi
v1 = 3.86

    