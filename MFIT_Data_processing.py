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
os.chdir('G:/My Drive/Core Flooding Project/MFIT_Rad_BTC/MFIT_Datafiles_V2')
#import the data process module
import data_process

# %% #decay correct the raw rad data
#run the decay correction with the data_process module from Chris Zahasky
filename = r"G:\My Drive\Core Flooding Project\UW Madison\Rad_BTC\Rad_data\18May2023_mg1771781c_PET_rad.csv"
s = data_process.sensors(filename, isotope = 'f18')
#check the starting experiment times, plot them and establish the x values
s.experiment_time_extraction(n=3, plot_check='yes')
#save the experiment starting times (rad 2) for each of the experiments
exp_start = s.eidx

exp_start[2] = 5560 #the 3rd experiment has super low radioactivity, so the auto detect doesnt work well
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
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6), sharex=True, gridspec_kw={'hspace': 0.5})

ax1.plot(s.minutes, s.SP1_raw)
ax1.set_title('Pressure data SP.1')
ax1.set_xlabel('Time [min]')
ax1.set_ylabel('Differential pressure [bars]')

ax2.plot(s.minutes, s.SP2_raw)
ax2.set_title('Pressure data SP.2')
ax2.set_xlabel('Time [min]')
ax2.set_ylabel('Differential pressure [bars]')

plt.show()
# %%
#test the above to see what the .csv data looks like
import pandas as pd
y = pd.read_csv('C1_pre_exp1.csv')
y = np.array(y.iloc[:,1])
x = np.arange(0, len(y))
plt.plot(x,y)
plt.xlim(0,100)

#attempting to do a baseline subtraction to fix the data
baseline = np.mean(y[0:15])
y_fix = y - baseline
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

'''BELOW IS ALL INFORMATION THAT IS PERTINANT TO AN ATTEMPT TO SMOOTH THE DATA'''
#%%
#deal with the bubble issue in the data, working at the individual curve scale
from scipy.interpolate import splrep, BSpline
#import the data and plot it in its initial form with bubbles present
#change the directory for the data
directory = 'G:/My Drive/Core Flooding Project/MFIT_Rad_BTC/Data'
os.chdir(directory)
file = 'C2_pre_exp3.csv'
#import the data from the datafile and load it into an array
data = np.loadtxt(file, delimiter=',', usecols=(0,1))
tck = splrep(data[:,0], data[:,1], s=0)
tck_s = splrep(data[:,0], data[:,1], s=130)


fig, ax = plt.subplots(1,1)
#plt.plot(data[:,0], data[:,1])
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
np.savetxt('C2_pre_exp3_fit.csv', b, delimiter=',')

# %%

'''Attempting a different method of smoothing using a lowess smoothing filter'''

import statsmodels.api as sm

#import some example data
file = r"G:\My Drive\Core Flooding Project\MFIT_Rad_BTC\Normalized_Data\RadBTC_2C_Post_009ml.csv"
a = np.loadtxt(file, delimiter=',')

x = a[:,0]
y = a[:,1]
# Generate some random data
#x = np.linspace(0, 10, 100)
#y = np.sin(x) + np.random.normal(0, 0.1, 100)

# Apply Lowess smoothing
lowess = sm.nonparametric.lowess(y, x, frac=.001)

# Plot the original data and the smoothed data
plt.plot(x, y)
plt.plot(lowess[:, 0], lowess[:, 1], c='r')
y_smooth = lowess[:,1]
export = np.hstack((x.reshape(-1, 1), y_smooth.reshape(-1, 1)))

fname = 'RadBTC_2C_Post_009ml_smooth.csv'
#np.savetxt(f'G:/My Drive/Core Flooding Project/MFIT_Rad_BTC/Normalized_Data/smooth/{fname}.csv', export, delimiter=',')


# %%
'''Attempting a different method of smoothing using a moving average filter'''
'''This does not work as it shifts the peak of the data and lowers the max value,
therefore the smoothing algoritm will have a substantial negative effect on the
moment analysis and MFIT inversions'''
import pandas as pd
#import some example data
file = r"G:\My Drive\Core Flooding Project\MFIT_Rad_BTC\Normalized_Data\RadBTC_1C_Post_06ml.csv"
data = pd.read_csv(file, header=None)
data['moving_average'] = data.iloc[:,1].rolling(window=22).mean()

plt.plot(data.iloc[:,0],data.iloc[:,2])
plt.plot(data.iloc[:,0], data.iloc[:,1])

#%%
'''Hand smoothing is not an option, it is not a robust methodology and it isnt accurate'''
'''Using a savgol_filter to smooth the data, exporting'''

from scipy.signal import savgol_filter

file = r"G:\My Drive\Core Flooding Project\MFIT_Rad_BTC\Normalized_Data\RadBTC_1C_Pre_6ml.csv"
a = np.loadtxt(file, delimiter=',')

x = a[:,0]
y = a[:,1]
y_filtered = savgol_filter(y, window_length=2, polyorder=1)

plt.plot(x, y, label='Noisy Data')
plt.plot(x, y_filtered, label='Filtered Data')
plt.legend()
plt.show()

residual = y-y_filtered
RMSE = np.sqrt(np.mean((y_filtered-y)**2))

export = np.hstack((x.reshape(-1, 1), y_filtered.reshape(-1, 1)))
plt.plot(x,residual)
plt.plot(x, RMSE)
fname = 'RadBTC_1C_Pre_6ml_smooth.csv'
#np.savetxt(f'G:/My Drive/Core Flooding Project/MFIT_Rad_BTC/Normalized_Data/smooth/{fname}.csv', export, delimiter=',')

#%%
'''This seems to be the best way to smooth the pre rxn, bubble impacted data without introducing
any undue bias through a curve fitting algorithm'''


from scipy.signal import medfilt
# prepare data
file = r"G:\My Drive\Core Flooding Project\MFIT_Rad_BTC\Normalized_Data\RadBTC_2C_pre_009ml.csv"
a = np.loadtxt(file, delimiter=',')


x = a[:,0]
y = a[:,1]


y_s = medfilt(y, 575)
fig1, ax1 = plt.subplots(dpi=400)
plt.plot(x,y, label='raw')
plt.plot(x,y_s, label='medfilt')
plt.legend(loc='best')
export = np.hstack((x.reshape(-1, 1), y_s.reshape(-1, 1)))

fname = 'RadBTC_1C_Pre_06ml_smooth.csv'
np.savetxt(f'G:/My Drive/Core Flooding Project/MFIT_Rad_BTC/Normalized_Data/smooth/{fname}.csv', export, delimiter=',')


# %%
'''Calculate and plot residuals for each of the simulations, as well as plot the 
phi values'''

from scipy import signal
#because the MFIT data only produces an array of 500 points, we need to downsample
#the observational data in order to get useable values for calculating residuals
###This may indicate a need to boost the number of values produced by the MFIT inversions!!!!!!!!

flo_rate = 0.09
xlim = (0,400) #change this to adjust the x lim for the different flow rates
#get the MFIT model output data (MDMi)
coreID = '1C_Pre'
directory = 'G:/My Drive/Core Flooding Project/MFIT_Rad_BTC/Normalized_Data/smooth/009mL_min/1C_Pre'
subfolder = 'MDMi'
filename = 'output.txt'

path2data = os.path.join(directory, subfolder)

os.chdir(path2data)
outMDMi = np.loadtxt(filename, skiprows=1, delimiter=';')
MDMi_x = outMDMi[:,0]
MDMi_C = outMDMi[:,1]
phiMDMi = np.loadtxt('Phi(N).txt', skiprows = 1)
#get the MFIT model output data (SFDM)
subfolder = 'SFDM'
filename = 'output.txt'

path2data = os.path.join(directory, subfolder)
os.chdir(path2data)
outSFDM = np.loadtxt(filename, skiprows=1, delimiter=';')
SFDM_x = outSFDM[:,0]
SFDM_C = outSFDM[:,1]
phiSFDM = np.loadtxt('Phi(N).txt', skiprows = 1)
#why on earth is the 2RNE data the same as the MDMi data???
#get the MFIT model output data (2RNE)
subfolder = '2RNE'
filename = 'output.txt'

path2data = os.path.join(directory, subfolder)
os.chdir(path2data)
outRNE= np.loadtxt(filename, skiprows=1, delimiter=';')
RNE_x = outRNE[:,0]
RNE_C = outRNE[:,1]
phi2RNE = np.loadtxt('Phi(N).txt', skiprows = 1)
#get the observational data
directory = 'G:/My Drive/Core Flooding Project/MFIT_Rad_BTC/Normalized_Data/smooth/009mL_min/1C_Pre'
filename = 'RadBTC_1C_Pre_009ml_smooth.csv.csv'

os.chdir(directory)
obs = np.loadtxt(filename, delimiter=',', usecols=(0,1))
obs = signal.resample(obs[:,1], 500) #downsampling the observational data to only 500 points in order to get residuals. this indicates that I
#probably need to go through and make sure that I have increased the number of points output by MFIT
#create the residuals array, plot them
#MDMi
rMDMi = np.subtract(MDMi_C, obs)
#SFDM
rSFDM = np.subtract(SFDM_C, obs)
#2RNE
r2RNE = np.subtract(RNE_C, obs)

fig3, ax3 = plt.subplots(figsize=(12,9), dpi = 200)
plt.plot(MDMi_x, rMDMi, label='MDMi Residual')
plt.plot(SFDM_x, rSFDM, label='SFDM Residual')
plt.plot(RNE_x, r2RNE, label='2RNE Residual')
plt.legend(loc='best')



#calculate the RMSE
RMSE_MDMi = np.linalg.norm(rMDMi) / np.sqrt(len(rMDMi))
RMSE_SFDM = np.linalg.norm(rSFDM) / np.sqrt(len(rSFDM))
RMSE_2RNE = np.linalg.norm(r2RNE) / np.sqrt(len(r2RNE))
#include this on the Residuals plot
plt.text(200, .05, f'RMSE_MDMi : {RMSE_MDMi:.4f}\nRMSE_SFDM: {RMSE_SFDM:.4f}\nRMSE_2RNE: {RMSE_2RNE:.4f}',
         fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

