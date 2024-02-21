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
directory = 'D:\COMSOL_Files\COMSOL_BTC_DATA'
os.chdir(directory)

#import the file 
#import the data and store it as a pandas dataframe
filename = 'adv_geom_BTC_test_6ml_v2_10sec.txt'
step = 1 #give the step size in seconds (different for different experiments and different export setups)
data = pd.read_table(filename, header=9, sep=r'\s+',
                     names=['x', 'y', 'z', 'c (mol/m^3) @ t=0', 'c (mol/m^3) @ t=1',
                            'c (mol/m^3) @ t=2','c (mol/m^3) @ t=3','c (mol/m^3) @ t=4',
                            'c (mol/m^3) @ t=5','c (mol/m^3) @ t=6', 'c (mol/m^3) @ t=7',
                            'c (mol/m^3) @ t=8','c (mol/m^3) @ t=9']) #clearly this needs to be automated based on the timestep
# %%
#generate an arrays of the 'total' concentration across the plane by calculating
#the concentration at every single point and summing them
#totcon_array through time
totcon_array = np.zeros(len(data.columns)-3)
#loop through and get the total concentration exiting the domain at each timestep
for i in range(3,len(data.columns)): #ignore the x, y, z columns
    ts = i-3 #time in seconds in the simulation
    totcon = data.iloc[:, i].sum() #aggregate the total concentration at all points
    totcon_array[ts] = totcon
    
#now plot this data on a timeseries graph with the x-axis as time
x = np.arange(0, len(totcon_array), step) #build the time component for the BTC

plt.plot(x, totcon_array)

# %%
# Initialize an empty list to store the strings
strings_list = []

# Define the initial value for t
t_value = 0.0

# Define the increment value for t
increment = 0.05

# Define the stop value for t
stop_value = 18.75 #will stop 

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
        
        # Format the string with the rounded t_value
        string = f'c (mol/m^3) @ t={rounded_t_value}'
    
    # Append the string to the list
    strings_list.append(string)
    
    # Update the value of t for the next iteration
    t_value += increment

# Display the list of strings
print(strings_list)