# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:56:21 2023

@author: csouc
"""

"""This script processes decay corredted data (using data_process.py) for import
into MFIT-BTC for transport process analysis, and then is used to take the output
data and plot it against the observational data"""

#import the necessary package dependencies
import numpy as np
from matplotlib import pyplot as plt
import os

#import the data_process python file and run it to decay correct the data for MFIT import
os.chdir('G:/My Drive/Core Flooding Project/UW Madison/Rad_BTC')
#import the data process module
import data_process

# %% #decay correct the raw rad data
#run the decay correction with the data_process module from Chris Zahasky
filename = r"G:\My Drive\Core Flooding Project\UW Madison\Rad_BTC\Rad_data\10May2023_mg1771781c_PET_rad.csv"
s = data_process.sensors(filename, isotope = 'f18')
#check the starting experiment times, plot them and establish the x values
s.experiment_time_extraction(n=3, plot_check='yes')
#save the experiment starting times (rad 2) for each of the experiments
exp_start = s.eidx

exp_start[2] = 8370 #the 3rd experiment has super low radioactivity, so the auto detect doesnt work well
#manually establish the 3rd data point here
exp_start = np.insert(exp_start, 3, len(s.SR3_dc))
print(exp_start) #check these values to see that they match the plots
# %%
#plot the data from the 3rd radioactivity sensor to see the total SR3 data collected
x = np.linspace(0, len(s.SR3_dc)-1,num=len(s.SR3_dc))
plt.plot(x, s.SR3_dc)

#use the start time data to export each of the experiments as .csv files for MFIT
#set the number of experiments
n=3
#name the core and the before/after indicator
ident = 'C1_pre'
#run a loop to extract the appropriate data
for i in range(len(s.SR3_dc)):
    if i <= 2:
        t_min = exp_start[i]
        t_max = exp_start[i+1]
        arr = s.SR3_dc[t_min:t_max]
        time = np.arange(arr.shape[0])
        arr = np.column_stack((time, arr))# add a time array
        weight = np.ones_like(time)
        arr=np.column_stack((arr, weight))
        #save the array as a .csv file
        fname = f'\\{ident}_exp{i+1}.csv'
        loc = os.getcwd()
        #np.savetxt(loc+fname, arr, delimiter=',')
        #print(np.max(arr))
        
    else:
        break
# %%
#test the above to see what the .csv data looks like
import pandas as pd
y = pd.read_csv('C1_pre_exp1.csv')
x = np.linspace(0, len(y)-1, len(y))
plt.plot(x,y)
#%%
#integrate under the curve to get the total tracer mass to input in MFIT
mass = np.trapz(y.iloc[:, 1])
# %%
#MFIT doesn't work with data values that are negative or 0, setting the lower limit
#of the data to 0.0000001
#for i in range(len(data)):
#    if data[i,1] <=  0.0000001:
#        data[i,1] =  0.0000001
#plot a test plot of the raw observational data
#fig =plt.subplot()
#plt.plot(data[:,0], data[:,1])
#plt.title('Timeseries plot of Rad BTC data')
#plt.xlabel('Time in Seconds')
#plt.ylabel('Radioconcentration [mCi/mL]')

#%%
#deal with the bubble issue in the data, working at the individual curve scale
from scipy.interpolate import splrep, BSpline
#import the data and plot it in its initial form with bubbles present
#change the directory for the data
directory = 'G:/My Drive/Core Flooding Project/MFIT_Rad_BTC/Data'
os.chdir(directory)
file = 'C1_pre_exp3.csv'
#import the data from the datafile and load it into an array
data = np.loadtxt(file, delimiter=',', usecols=(0,1))
tck = splrep(data[:,0], data[:,1], s=0)
tck_s = splrep(data[:,0], data[:,1], s=11)


fig, ax = plt.subplots(1,1)
plt.plot(data[:,0], data[:,1])
#plt.plot(data[:,0],BSpline(*tck)(data[:,0]))
plt.plot(data[:,0],BSpline(*tck_s)(data[:,0]))
plt.title('0.09 mL/min BTC')
plt.xlabel('Time (seconds)')
plt.ylabel('Radioconcentration (decay corrected)')

# replace the data with the fitted data and export it as a new .csv
new = BSpline(*tck_s)(data[:,0])
np.min(new)
b = np.column_stack((np.arange(len(new)),new, np.ones(len(new))))

#check that this worked
#save this as a csv file
np.savetxt('C1_pre_exp3_fit.csv', b, delimiter=',')

