# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:23:29 2024

@author: Charlie
"""
'''Testing to see if I can create a good BTC from the exported comsol data'''

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
#change directory 
#directory = 'D:\COMSOL_Files\COMSOL_BTC_DATA' #desktop directory
directory = r'C:\Users\csouc\Core Flooding Laptop\COMSOL Simulations\COMSOL_BTC_data' #laptop directory
os.chdir(directory)


# Initialize an empty list to store the strings
strings_list = []

# Define the initial value for t
t_value = 0.0

# Define the increment value for t
increment = 0.1

# Define the stop value for t
stop_value = 300

# Define the number of decimal places to round
decimal_places = 2

# Loop to generate the strings
while t_value <= stop_value:
    # Check if the t_value is exactly 0.0
    if t_value == 0.0:
        # Format the string without decimal places
        string = f'c (mol/m^3) @ t=0'
    else:
        # Round the t_value to the desired number of decimal places
        rounded_t_value = round(t_value, decimal_places)
        
        # If the rounded value is a whole number or has only one digit after the decimal point
        if rounded_t_value.is_integer() or rounded_t_value % 1 == 0.0:
            # Format the string without the decimal part
            string = f'c (mol/m^3) @ t={int(rounded_t_value)}'
        else:
            # Format the string with the rounded t_value
            string = f'c (mol/m^3) @ t={rounded_t_value}'
    
    # Append the string to the list
    strings_list.append(string)
    
    # Update the value of t for the next iteration
    t_value += increment

    # Handle floating-point approximation issues
    if round(t_value, 2) > 300:
        break

# Insert the x, y, and z column headers
strings_list.insert(0, 'x')
strings_list.insert(1, 'y')
strings_list.insert(2, 'z')

#import the file 
#import the data and store it as a pandas dataframe
filename = 'adv_geom_BTC_test_009ml_v2_frac.txt'
step = 0.1 #give the step size in seconds (different for different experiments and different export setups)
data = pd.read_table(filename, header=9, sep=r'\s+',
                     names=strings_list, index_col=False) #clearly this needs to be automated based on the timestep
# %%
#generate an arrays of the 'total' concentration across the plane by calculating
#the concentration at every single point and summing them
#totcon_array through time
savefig = False
totcon_array = np.zeros(len(data.columns)-3)
#loop through and get the total concentration exiting the domain at each timestep
for i in range(3,len(data.columns)): #ignore the x, y, z columns
    ts = i-3 #time in seconds in the simulation
    totcon = data.iloc[:, i].sum() #aggregate the total concentration at all points
    totcon_array[ts] = totcon
    
#now plot this data on a timeseries graph with the x-axis as time
x = np.arange(0, 300.1, step) #build the time component for the BTC

#plot the breakthrough curve data
fig, ax = plt.subplots(figsize=(12,9), dpi = 800)
plt.plot(x, totcon_array, linewidth=4)
plt.xlabel('Time [seconds]', fontsize=24)
plt.ylabel('Concentration [mol/m^3]', fontsize=24)
plt.title('COMSOL 0.09mL/min BTC Fracture', fontsize=30)
ax.tick_params(axis='both', labelsize=20)
if savefig:
    #change the directory to a figure directory
    savedirectory = r'C:\Users\csouc\Core Flooding Laptop\Plots'
    os.chdir(savedirectory)
    plt.savefig('009mL_frac_BTC.png', dpi=800)
    directory = r'C:\Users\csouc\Core Flooding Laptop\COMSOL Simulations\COMSOL_BTC_data' #laptop directory
    os.chdir(directory)