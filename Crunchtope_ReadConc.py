# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:26:12 2023

@author: csouc
"""
"""This is a python code designed to take CrunchTope Concentration and
concentration history files and plot them over distance and time"""
#Import the necessary package dependencies
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

# %%
#Set the appropriate working directory for the modeling. This should contain
# the input files, database files, .ant files, and the output files

#change this path to the appropriate directory before running
path = 'C:/Users/csouc/RTM_Workshop/Preliminary_Core_Flooding_RTM'
os.chdir(path)

# %%
'''Set the basics for the number of files and the number of cells'''

nfiles = 0

prefixes = ['conc', 'totcon', 'totconhistory']
#read through to determine how many concentration files there are that match
#the above prefixes

for i in range nfiles:
    