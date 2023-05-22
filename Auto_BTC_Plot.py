# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:30:24 2023

@author: csouc
"""

"""This code is designed to take data from breakthrough curve CSV files
and plot them with normalization based on the area under the curve.
This code also has the ability to conduct moment analysis to calculate
the porosity of the medium"""

"""For the best results, consolidate data onto a single .csv file"""

"""Much of the setup here is designed to work with the BTC data collected
at UW Madison during our tracer tests on 08May2023 and 17May2023, preceeding
our PET scanning. While lots of tracer data is imported, only a few of
the flow rates are actually considered for plotting, as they were conducted
across cores both before and after the dissolution experiments. Most of the
data adjustments are made outside of the plot_by_flow function as it is
just easier to make small adjustments on the fly that way"""


"""Naming convention for columns in the input csv file is as follows:
    DD:Mon:YYYY_tracer_#C_XmLpermin Ex: 08May2023_tracer_1C_6mLpermin
    if the flow rate is a decimal value, use the following format:
        DD:Mon:YY_tracer_#C_pXmlpermin    17May2023_tracer_1C_p6mlpermin
        where the p represents the decimal in the flow rate value"""

"""Be sure that this python file is in the same folder as the data that 
you wish to analyze and plot"""

"""In order to edit this code to use for other plotting purposes, make
the necessary changes as outlined below. These outlines show the areas that
are specific to the UW flouresciein tracer data."""

"""This code cannot be used for radioactive BTC data as it does not have 
any decay correction code."""

#Import package dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import inspect
# %%
'''Define the functions to access and plot the data from the raw_data
dataframe'''
def split_pre_post(df):
    pre_conc = []    #these two empty lists are specific to the data setup
    post_conc = []  #of the UW breakthrough curve data
    for column in df:
        if column[0] == '0':   #this is based on the date info in the datafiles, 08May or 17May
            pre_conc.append(df[column])
        elif column[0] == '1':
            post_conc.append(df[column])
    return(pre_conc, post_conc)


#eliminate and drop all of the nan values from a series of dataframes
def remove_nan_values(series_list):
    cleaned_series = []
    for series in series_list:
        cleaned_series.append(series.dropna())
    return cleaned_series


#This is a necessary tool that is currently set up specificially in order
#to work with the UW Breakthrough curve data, as it looks for specific
#strings in the code setup, which need to be adjusted for different purposes
#ideally, chanigng the pattern variable to stratify your data would be ideal
def group_series_by_flow_rate(series_list):
    dataframes = {}
    pattern = r'_tracer_(\d+[C|c])_(p?\d+(?:\.\d+)?)(m[lL])permin$'

    for series in series_list:
        match = re.search(pattern, series.name, re.IGNORECASE)

        if match: #this match value is linked to the pattern above, separates keywords by core. Will need alteration
            core = match.group(1).upper()  # Convert core value to upper case
            flow_rate = match.group(2)
            unit = match.group(3).lower()  # Convert unit value to lower case

            if flow_rate.startswith('p'):  #this is used to separate decimal flow rate from integer flow rate (e.g. 6mL/,min vs 0.6mL/min)
                flow_rate = flow_rate[1:]  # Remove the 'p' prefix for decimal flow rates
                decimal_key = f"p{flow_rate}{unit}"
            else:
                decimal_key = f"{flow_rate}{unit}"

            if decimal_key not in dataframes:
                dataframes[decimal_key] = {}

            if core not in dataframes[decimal_key]:
                dataframes[decimal_key][core] = pd.DataFrame(series)
            else:
                dataframes[decimal_key][core] = pd.concat([dataframes[decimal_key][core], pd.DataFrame(series)], axis=1)

    return dataframes


#Get the name of the dictionary to use in the plot title
def get_list_name(lst):
    frame = inspect.currentframe()
    for name, obj in frame.f_back.f_globals.items():
        if obj is lst:
            return name

#this will only plot for the 1mL/min, 6mL/min, and 0.6mL/min
#flow rates, as these are ones that we conducted both before and
#after the dissolution experiments. This will also need alteration to create
# the appropriate dataframes based on your own input files
def group_by_flow(prediss, postdiss):
    #initialize empty dataframes for each of the flow rates
    sixmL_min = []
    onemL_min = []
    point6mL_min = []
    #6mL/min
    sixmL_min.append(prediss.get('6ml'))
    sixmL_min.append(postdiss.get('6ml'))
    #1mL/min
    onemL_min.append(prediss.get('1ml'))
    onemL_min.append(postdiss.get('1ml'))
    #0.6mL/min
    point6mL_min.append(prediss.get('p6ml'))
    point6mL_min.append(postdiss.get('p6ml'))
   
    return(sixmL_min, onemL_min,point6mL_min)


#pore volume adjustment occurs later, not in this function
#needs to also have normalization applied
def plot_by_flow(lst,flo):
    for i in range(len(lst)):
        d = lst[i]  #d is a dictionary within the input list
        keyslist = d.keys()
        title = get_list_name(lst)
        for key in keyslist:
            #need to implement the normalization and pore volumes 
            df = d.get(key)
            #print(df.iloc[:,0])
            label = str(df.iloc[0])
            label = label[0:29]
            if label == '08May2023_tracer_2C_p6mlpermi':
                integral = np.trapz(df.iloc[:4000,0])
            else:
                integral = np.trapz(df.iloc[:,0])
            #print(integral)
            tinj = 1/ flo *60
            C0 = integral/tinj
            #integrate for normalization
            plt.plot(np.arange(0,len(d.get(key))),d.get(key)/C0, label = label)
            plt.legend(loc='best')
            plt.title(f'UV-Vis BTC at {title} for cores 1C, 2C')
            plt.xlabel('Pore Volumes')
            plt.ylabel('C/C0')
         
            
         
# used for test purposes in order to determine if the labels being created
#above are as expected based on the file naming conventions
def test_label(lst):
    for i in range(len(lst)):
        d = lst[i]
        keyslist = d.keys()
        for key in keyslist:
            df = d.get(key)
            label =str(df.iloc[0])
            print(label)
            
#need to also create a function to do moment analysis, could be used in 
#order to determine pore volume calculations

# %%
"""Import master csv with all of the Breakthrough curve data. For
different files be sure to change this"""
##May come back and make this an automated import system so that
#this can be run purely from the command prompt

os.chdir(r'C:\Users\csouc\OneDrive\Documents\Core Flooding Project\UW Madison')
raw_data = pd.read_csv('UW_MG177178_Core_Tracers_MASTER.csv')
# %%
'''get all of the preflood and post flooddataframes gathered'''
preflood, postflood = split_pre_post(raw_data)
preflood = remove_nan_values(preflood)
postflood = remove_nan_values(postflood)

# %%
'''Group all of the series into pre and post flood data'''
prediss=group_series_by_flow_rate(preflood)
postdiss = group_series_by_flow_rate(postflood)
# %%
'''Plot all of the data. Each plot should contain the breakthrough
curves from the same flow rate, both before and after the dissolution
experiments are conducted'''


'''There has been some question regarding which pore volume (PVx) is best
to use for which core. PV1 comes from the original shapefile fracture vol.
while PV2 and PV3 were calculated from the initial breakthrough curves using
the time between injection and breakthrough, multipled by flow rate and converted
to a volume. For now, all PV values are being used with the PV1 value, as 
it allows for uniform normalization based on the experiments pre dissolution'''
sixmL_min, onemL_min, point6mL_min = group_by_flow(prediss, postdiss)

# %%
"""Below are all the plots for the different traces at the different flow
rates. This is based on the structure of the dataframes output by flow
rate, and is plotted in both log and semilog plots. There is also an x axis
converseion conducted in order to transition from units of time in seconds
to units of pore volume."""
# ****As of 5/22/23, we are utilizing only the data for PV1, calculated
# from the pore volume from the CAD file. This is to keep everything uniform
#along the x-axis in order to conduct the appropriate comparisons. #this may
# be changed at some point in the future to a different pore volume normalization


"""The code for these plots will look very different depending on the type
of data that you are working with and how you have it stratified. The general
theme is to plot all of the data from the appropriate dataframes and then use
the pore volumes to reconstruct the x-axis.Note that these pore volume calculations
are conducted in the form original x values/PV *florate, which is then passed
back into each of the curves separately"""
# %%
'''6mL/min non-log'''

#set before and after dissolution measurements for pore volumes
#these pore volume calculations are including tubing deadvol.
PV1 = .89946
PV2 = 1.425
PV3 = 1.245
fig1, ax1 = plt.subplots()
plot_by_flow(sixmL_min,6)
plt.xlim(0,6.5)

#alter this to be in pore volumes
lines = plt.gca().get_lines()  # Get all Line2D objects
curve1 = lines[0]  # Get the Line2D object for Curve 1
x_values_curve1 = curve1.get_xdata()  # Get x-axis values for Curve 1

curve2 = lines[1]
x_values_curve2 = curve2.get_xdata()

curve3 = lines[2]
x_values_curve3 = curve3.get_xdata()

newxcurve1 = x_values_curve1/PV1 * 6/60
newxcurve2 = x_values_curve2/PV1 * 6/60
newxcurve3 = x_values_curve3/PV1 * 6/60


lines1 = plt.gca().get_lines()[0]  # Get the Line2D object for Curve 1
lines1.set_xdata(newxcurve1)

lines2 = plt.gca().get_lines()[1]  # Get the Line2D object for Curve 1
lines2.set_xdata(newxcurve2)

lines3 = plt.gca().get_lines()[2]  # Get the Line2D object for Curve 1
lines3.set_xdata(newxcurve3)


plt.savefig('6mL_min.png', dpi=500)
# %%
'''1mL/min non-log'''

#set before and after dissolution measurements for pore volumes
PV1 = .89946
PV2 = 1.425
PV3 = 1.245
fig2, ax2 = plt.subplots()
plot_by_flow(onemL_min,1)
plt.xlim(0,7)

#alter this to be in pore volumes
lines = plt.gca().get_lines()  # Get all Line2D objects
curve1 = lines[0]  # Get the Line2D object for Curve 1
x_values_curve1 = curve1.get_xdata()  # Get x-axis values for Curve 1

curve2 = lines[1]
x_values_curve2 = curve2.get_xdata()

curve3 = lines[2]
x_values_curve3 = curve3.get_xdata()

curve4 = lines[3]
x_values_curve4 = curve4.get_xdata()

newxcurve1 = x_values_curve1/PV1 * 1/60
newxcurve2 = x_values_curve2/PV1 * 1/60
newxcurve3 = x_values_curve3/PV1 * 1/60
newxcurve4 = x_values_curve4/PV1 * 1/60

lines1 = plt.gca().get_lines()[0]  # Get the Line2D object for Curve 1
lines1.set_xdata(newxcurve1)

lines2 = plt.gca().get_lines()[1]  # Get the Line2D object for Curve 1
lines2.set_xdata(newxcurve2)

lines3 = plt.gca().get_lines()[2]  # Get the Line2D object for Curve 1
lines3.set_xdata(newxcurve3)

lines4 = plt.gca().get_lines()[3]  # Get the Line2D object for Curve 1
lines4.set_xdata(newxcurve4) 

plt.savefig('1mL_min.png', dpi=500)

# %%
'''0.6mL/min non-log'''

#set before and after dissolution measurements for pore volumes
PV1 = .89946
PV2 = 1.425
PV3 = 1.245
fig3, ax3 = plt.subplots()
plot_by_flow(point6mL_min, 0.6)
plt.xlim(0,10)

#alter this to be in pore volumes
lines = plt.gca().get_lines()  # Get all Line2D objects
curve1 = lines[0]  # Get the Line2D object for Curve 1
x_values_curve1 = curve1.get_xdata()  # Get x-axis values for Curve 1

curve2 = lines[1]
x_values_curve2 = curve2.get_xdata()

curve3 = lines[2]
x_values_curve3 = curve3.get_xdata()

newxcurve1 = x_values_curve1/PV1 * 0.6/60
newxcurve2 = x_values_curve2/PV1 * 0.6/60
newxcurve3 = x_values_curve3/PV1 * 0.6/60


lines1 = plt.gca().get_lines()[0]  # Get the Line2D object for Curve 1
lines1.set_xdata(newxcurve1)

lines2 = plt.gca().get_lines()[1]  # Get the Line2D object for Curve 1
lines2.set_xdata(newxcurve2)

lines3 = plt.gca().get_lines()[2]  # Get the Line2D object for Curve 1
lines3.set_xdata(newxcurve3)

plt.savefig('point6mL_min.png', dpi=500)



# %%
'''6mL/min semilog'''

PV1 = .89946
PV2 = 1.425
PV3 = 1.245
#now do this in log scale
fig4, ax4 = plt.subplots()
plot_by_flow(sixmL_min, 6)
plt.xlim(0,6.5)
plt.yscale('log')

#alter this to be in pore volumes
lines = plt.gca().get_lines()  # Get all Line2D objects
curve1 = lines[0]  # Get the Line2D object for Curve 1
x_values_curve1 = curve1.get_xdata()  # Get x-axis values for Curve 1

curve2 = lines[1]
x_values_curve2 = curve2.get_xdata()

curve3 = lines[2]
x_values_curve3 = curve3.get_xdata()

newxcurve1 = x_values_curve1/PV1 *6/60
newxcurve2 = x_values_curve2/PV1 * 6/60
newxcurve3 = x_values_curve3/PV1 * 6/60


lines1 = plt.gca().get_lines()[0]  # Get the Line2D object for Curve 1
lines1.set_xdata(newxcurve1)

lines2 = plt.gca().get_lines()[1]  # Get the Line2D object for Curve 1
lines2.set_xdata(newxcurve2)

lines3 = plt.gca().get_lines()[2]  # Get the Line2D object for Curve 1
lines3.set_xdata(newxcurve3)

plt.savefig('6mL_min_semilog.png', dpi=500)



# %%
'''1mL/min semilog'''


PV1 = .89946
PV2 = 1.425
PV3 = 1.245
fig5, ax5 = plt.subplots()
plot_by_flow(onemL_min, 1)
plt.xlim(0,7)
plt.yscale('log')

#alter this to be in pore volumes
lines = plt.gca().get_lines()  # Get all Line2D objects
curve1 = lines[0]  # Get the Line2D object for Curve 1
x_values_curve1 = curve1.get_xdata()  # Get x-axis values for Curve 1

curve2 = lines[1]
x_values_curve2 = curve2.get_xdata()

curve3 = lines[2]
x_values_curve3 = curve3.get_xdata()

curve4 = lines[3]
x_values_curve4 = curve4.get_xdata()

newxcurve1 = x_values_curve1/PV1 * 1/60
newxcurve2 = x_values_curve2/PV1 * 1/60
newxcurve3 = x_values_curve3/PV1 * 1/60
newxcurve4 = x_values_curve4/PV1 * 1/60

lines1 = plt.gca().get_lines()[0]  # Get the Line2D object for Curve 1
lines1.set_xdata(newxcurve1)

lines2 = plt.gca().get_lines()[1]  # Get the Line2D object for Curve 1
lines2.set_xdata(newxcurve2)

lines3 = plt.gca().get_lines()[2]  # Get the Line2D object for Curve 1
lines3.set_xdata(newxcurve3)

lines4 = plt.gca().get_lines()[3]  # Get the Line2D object for Curve 1
lines4.set_xdata(newxcurve4) 


plt.savefig('1mL_min_semilog.png', dpi=500)


# %%
'''0.6mL/min semilog'''


PV1 = .89946
PV2 = 1.425
PV3 = 1.245
fig6, ax6 = plt.subplots()
plot_by_flow(point6mL_min, 0.6)
plt.xlim(0,10)
plt.yscale('log')

#alter this to be in pore volumes
lines = plt.gca().get_lines()  # Get all Line2D objects
curve1 = lines[0]  # Get the Line2D object for Curve 1
x_values_curve1 = curve1.get_xdata()  # Get x-axis values for Curve 1

curve2 = lines[1]
x_values_curve2 = curve2.get_xdata()

curve3 = lines[2]
x_values_curve3 = curve3.get_xdata()

newxcurve1 = x_values_curve1/PV1 * 0.6/60
newxcurve2 = x_values_curve2/PV1 * 0.6/60
newxcurve3 = x_values_curve3/PV1 * 0.6/60


lines1 = plt.gca().get_lines()[0]  # Get the Line2D object for Curve 1
lines1.set_xdata(newxcurve1)

lines2 = plt.gca().get_lines()[1]  # Get the Line2D object for Curve 1
lines2.set_xdata(newxcurve2)

lines3 = plt.gca().get_lines()[2]  # Get the Line2D object for Curve 1
lines3.set_xdata(newxcurve3)

plt.savefig('point6mL_min_semilog.png', dpi=500)

