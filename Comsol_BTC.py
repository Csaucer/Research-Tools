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
directory = 'D:\COMSOL_Files\COMSOL_BTC_DATA\COMSOL_concentration_distributions' #desktop directory
#directory = r'C:\Users\csouc\Core Flooding Laptop\COMSOL Simulations\COMSOL_BTC_data' #laptop directory
os.chdir(directory)
# %%

# Initialize an empty list to store the strings
strings_list = []

# Define the initial value for t
t_value = 0.0

# Define the increment value for t
increment = 0.2

# Define the stop value for t
stop_value = 18.6

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
    if round(t_value, 2) > stop_value:
        break

# Insert the x, y, and z column headers
strings_list.insert(0, 'x')
strings_list.insert(1, 'y')
strings_list.insert(2, 'z')

#import the file 
#import the data and store it as a pandas dataframe
filename = 'adv_geom_fracture_C_6mL.txt'
step = 0.5 #give the step size in seconds (different for different experiments and different export setups)
data = pd.read_table(filename, header=9, sep=r'\s+',
                     names=strings_list, index_col=False) #clearly this needs to be automated based on the timestep
# %%
#generate an arrays of the 'total' concentration across the plane by calculating
#the concentration at every single point and summing them
#totcon_array through time
##THIS IS THE WRONG WAY TO DO THIS SEE THE BELOW SELECTION FOR THE PROPER WAY
#THIS JUST USES THE POINT CONCENTRATION DATA, NO CONSIDERATION OF VOXEL VOLUMES
savefig = False
totcon_array = np.zeros(len(data.columns)-3)
#loop through and get the total concentration exiting the domain at each timestep
for i in range(3,len(data.columns)): #ignore the x, y, z columns
    ts = i-3 #time in seconds in the simulation
    totcon = data.iloc[:, i].sum() #aggregate the total concentration at all points
    totcon_array[ts] = totcon
    
#now plot this data on a timeseries graph with the x-axis as time
x = np.arange(0, 1247, step) #build the time component for the BTC

#plot the breakthrough curve data
fig, ax = plt.subplots(figsize=(12,9), dpi = 800)
plt.plot(x, totcon_array, linewidth=4)
plt.xlabel('Time [seconds]', fontsize=24)
plt.ylabel('Concentration [mol/m^3]', fontsize=24)
plt.title('COMSOL 0.09mL/min BTC Channel', fontsize=30)
ax.tick_params(axis='both', labelsize=20)
if savefig:
    #change the directory to a figure directory
    #savedirectory = r'C:\Users\csouc\Core Flooding Laptop\Plots' #laptop directory
    savedirectory = 'D:\COMSOL_Files\COMSOL_BTC_Plots'#desktop directory
    os.chdir(savedirectory)
    plt.savefig('009mL_chan_BTC.png', dpi=800)
    #directory = r'C:\Users\csouc\Core Flooding Laptop\COMSOL Simulations\COMSOL_BTC_data' #laptop directory
    directory = 'D:\COMSOL_Files\COMSOL_BTC_DATA' #desktop directory
    os.chdir(directory)
    
# %% Generate a BTC plot for the total system, then a BTC of the Fracture, and of the Channel
#Generate Three Breakthrough Curves

savefig = False #flag for saving the figures
#6mL/min
florate = 6
#load BTC data
BTC =  pd.read_csv('total_outlet_C_6mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
frac_BTC =  pd.read_csv('fracture_outlet_C_6mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
chan_BTC =  pd.read_csv('channel_outlet_C_6mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
x = BTC.iloc[:,0] #get the time array in seconds
fig1, axs = plt.subplots(3,1 , figsize=(12,9))
fig1.suptitle('COMSOL BTC; 6mL/min', fontsize=20)
xlabel = 'Time [s]'
ylabel = 'Solute Amount [mols]' #the y axis is in mols as this is a volume integration

#plot the overall BTC 
axs[0].plot(x, BTC.iloc[:,1])
axs[0].set_title('Total BTC', fontsize=16)
axs[0].set_xlabel(xlabel, fontsize=14)
axs[0].set_ylabel(ylabel, fontsize=14)


#plot the fracture component of the BTC
axs[1].plot(x, frac_BTC.iloc[:,1])
axs[1].set_title('Fracture Outlet BTC', fontsize=16)
axs[1].set_xlabel(xlabel, fontsize=14)
axs[1].set_ylabel(ylabel, fontsize=14)



#plot the channel component of the BTC
axs[2].plot(x, chan_BTC.iloc[:,1])
axs[2].set_title('Channel Outlet BTC', fontsize=16)
axs[2].set_xlabel(xlabel, fontsize=14)
axs[2].set_ylabel(ylabel, fontsize=14)


plt.subplots_adjust(hspace=0.5, top=0.9) # Increase vertical spacing between subplots

if savefig:
    datadir = os.getcwd()
    savedir = 'D:\COMSOL_Files\COMSOL_BTC_Plots'
    os.chdir(savedir)
    plt.savefig(f'{florate}_mL_min_COMSOL_BTCs')
    os.chdir(datadir)
    
####### SEMILOG BTC's
fig2, axs2 = plt.subplots(3,1, figsize=(12,9))
fig2.suptitle('COMSOL BTC; 6mL/min Semilog', fontsize=20)
#do the same thing as above but with semilog y plots
#plot the overall BTC
axs2[0].plot(x, BTC.iloc[:,1], color='Orange')
axs2[0].set_title('Total BTC', fontsize=16)
axs2[0].set_xlabel(xlabel, fontsize=14)
axs2[0].set_ylabel(ylabel,fontsize=14)
axs2[0].semilogy()

#plot the fracture component of the BTC
axs2[1].plot(x, frac_BTC.iloc[:,1], color='Orange')
axs2[1].set_title('Fracture Outlet BTC', fontsize= 16)
axs2[1].set_xlabel(xlabel, fontsize=14)
axs2[1].set_ylabel(ylabel, fontsize=14)
axs2[1].semilogy()

#plot the channel component of the BTC
axs2[2].plot(x, chan_BTC.iloc[:,1], color='Orange')
axs2[2].set_title('Channel Outlet BTC', fontsize=16)
axs2[2].set_xlabel(xlabel, fontsize=14)
axs2[2].set_ylabel(ylabel, fontsize=14)
axs2[2].semilogy()

plt.subplots_adjust(hspace=0.5, top=0.9) # Increase vertical spacing between subplots

if savefig:
    datadir = os.getcwd()
    savedir = 'D:\COMSOL_Files\COMSOL_BTC_Plots'
    os.chdir(savedir)
    plt.savefig(f'{florate}_mL_min_semilog_COMSOL_BTCs.png')
    os.chdir(datadir)

#%%    
#get the data from the 0.09 and 6mL/min plotted on the same graph, normalized
#by time
#6mL/min
BTC_6 =  pd.read_csv('total_outlet_C_6mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
frac_BTC_6 =  pd.read_csv('fracture_outlet_C_6mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
chan_BTC_6 =  pd.read_csv('channel_outlet_C_6mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
x6 = BTC_6.iloc[:,0]
tstep6 = 0.05
#0.09mL/min
BTC_009 =  pd.read_csv('total_outlet_C_009mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
frac_BTC_009 =  pd.read_csv('fracture_outlet_C_009mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
chan_BTC_009 =  pd.read_csv('channel_outlet_C_009mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
x009 = BTC_009.iloc[:,0]
tstep009 = 0.5

 
#Total BTCs
#normalize the data by the peak time
#get the time of the data peaks
t1 = np.argmax(BTC_6.iloc[:,1]) * tstep6
t2 = np.argmax(BTC_009.iloc[:,1]) *tstep009

#x6 = x6/t1
#x009 = x009/t2
#plot the normalized breakthrough curves (normalized by peak arrival time)
#fig, ax = plt.subplots(figsize=(12,9), dpi=800)
#plt.plot(x6, BTC_6.iloc[:,1])
#plt.plot(x009, BTC_009.iloc[:,1])
#plt.semilogy()

#plot normalized by VI
x6 = BTC_6.iloc[:,0]
x009 = BTC_009.iloc[:,0] #reset to the original x axes


x6 = x6 * 100
x009 = x009 * 1.5
#total BTC
fig, ax = plt.subplots(figsize=(12,9), dpi=800)
plt.plot(x6, BTC_6.iloc[:,1], label='6mL/min')
plt.plot(x009, BTC_009.iloc[:,1], label='0.09mL/min')
plt.semilogy()
plt.title('VI Norm. BTC, Total Solute Discharge', fontsize=20)
plt.xlabel('Vol. Injected (mL)', fontsize=18)
plt.ylabel('Quantity of solute [mol]', fontsize=18)
plt.legend(loc='upper right', fontsize=18)
plt.ylim((10E-13,10E-7))
plt.tick_params(axis='x', which='major', labelsize=16) # x-axis tick size
plt.tick_params(axis='y', which='major', labelsize=16) # y-axis tick size

#fractures
fig, ax = plt.subplots(figsize=(12,9), dpi=800)
plt.plot(x6, frac_BTC_6.iloc[:,1], label='6mL/min')
plt.plot(x009, frac_BTC_009.iloc[:,1], label='0.09mL/min')
plt.title('VI Norm. BTC, Fracture Solute Discharge', fontsize=20)
plt.xlabel('Vol. Injected (mL)', fontsize=18)
plt.ylabel('Quantity of solute [mol]', fontsize=18)
plt.semilogy()
plt.legend(loc='upper right', fontsize=18)
plt.ylim((10E-13,10E-9))
plt.tick_params(axis='x', which='major', labelsize=16) # x-axis tick size
plt.tick_params(axis='y', which='major', labelsize=16) # y-axis tick size

#channels
fig, ax = plt.subplots(figsize=(12,9), dpi=800)
plt.plot(x6, chan_BTC_6.iloc[:,1], label='6mL/min')
plt.plot(x009, chan_BTC_009.iloc[:,1], label='0.09mL/min')
plt.title('VI Norm. BTC, Channel Solute Discharge', fontsize=20)
plt.xlabel('Vol. Injected (mL)', fontsize=18)
plt.ylabel('Quantity of solute [mol]', fontsize=18)
plt.legend(loc='upper right', fontsize=18)
plt.semilogy()
plt.ylim((10E-13,10E-7))
plt.tick_params(axis='x', which='major', labelsize=16) # x-axis tick size
plt.tick_params(axis='y', which='major', labelsize=16) # y-axis tick size

# %%
'''Using a similar process as outlined below with the % of total groundwater approach,
establish the % of total discharged concentration through the fracture, and the channel
throuhgout time'''
#first establish the total amount discharged throuhout the entire event
#and get % of solute discharged through the fracture and thru the channel
total_eff = np.trapz(BTC.iloc[:,1])
frac_eff = np.trapz(frac_BTC.iloc[:,1])
chan_eff = np.trapz(chan_BTC.iloc[:,1])
#Get the % discharged at the end of the model run
print(f'The % discharged through the fracture: {(frac_eff/total_eff)*100}')
print(f'The % discharged through the channel: {(chan_eff/total_eff)*100}')

frac_eff_per = (frac_BTC.iloc[:,1]/BTC.iloc[:,1])*100
chan_eff_per = (chan_BTC.iloc[:,1]/BTC.iloc[:,1])*100

#Now plot the % discharged throughout time
fig, ax = plt.subplots(figsize=(12,9), dpi=800)
plt.plot(x, chan_eff_per, label ='% discharge, channel', linewidth=3)
plt.plot(x, frac_eff_per, label= '% discharge, fracture', linewidth=3)
plt.legend(loc='center right', fontsize=16)
plt.ylabel('% of Total Tracer in System', fontsize=18)
plt.xlabel('Time [s]', fontsize=18)
plt.title('6mL/min Solute Discharge', fontsize=24)
plt.tick_params(axis='x', which='major', labelsize=16) # x-axis tick size
plt.tick_params(axis='y', which='major', labelsize=16) # y-axis tick size
plt.text(12.5, 35, f'% Total, Fracture: {(frac_eff/total_eff)*100:.2f}\n% Total, Channel: {(chan_eff/total_eff)*100:.2f}', fontsize=16, color='k', bbox=dict(facecolor='red', alpha=0.4))
os.chdir(savedir)
#plt.savefig('solute_discharge_6mL.png')
os.chdir(datadir)




# %%
# I may have established a better way to integrate the concentration data spatially
#in COMSOL. Plotting that data here

"""import the data for the fracture, channel, initial quantity (mol)
remember that this data is in mols as it is volume integrated
THIS IS NOT BTC DATA. That requires a surface (2D) integration at the outlet
of the system"""
savefig = False #flag operator for saving the figure as a .png
florate = 6

frac = pd.read_csv('fracture_C_6mL_integration_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
chan = pd.read_csv('Channel_C_6mL_integration_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
inlet = pd.read_csv('inlet_block_C_6mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])

#plot the total amount of solute in the fracture and the total solute in the channel, and the total solute overall (sum)
#the total amount of solute in the domain is going to change over time as solute
#leaves the domain. Hence, we need to sum the total amount of solute in each domain to generate a % of total solute in each domain
totals = inlet.iloc[:,1] + chan.iloc[:,1] + frac.iloc[:,1]
frac_per = (frac.iloc[:,1]/totals) * 100
chan_per = (chan.iloc[:,1]/totals) * 100
inlet_per = (inlet.iloc[:,1]/totals) * 100
x = frac.iloc[:,0] #time in seconds
#Plot of % of total concentrations
fig3, ax3 = plt.subplots(figsize=(12,9), dpi=800)
plt.plot(x, chan_per, label = '% in Channel', linewidth=3)
plt.plot(x, frac_per, label='% in Fracture', linewidth=3)
plt.plot(x, inlet_per, label="% in Inlet Box",linewidth=3, alpha= 0.3)
plt.title('6mL/min Solute Distribution', fontsize=24)
plt.ylabel('% of Total Tracer in System', fontsize=18)
plt.xlabel('Time [s]', fontsize=18)
plt.tick_params(axis='x', which='major', labelsize=16) # x-axis tick size
plt.tick_params(axis='y', which='major', labelsize=16) # y-axis tick size
plt.legend(loc='upper right', fontsize=16)
if savefig:
    datadir = os.getcwd()
    savedir = 'D:\COMSOL_Files\COMSOL_BTC_Plots'
    os.chdir(savedir)
    plt.savefig(f'{florate}mL_min_solute_distribution')
    os.chdir(datadir)
    
#%%
#commented out as I have already done this below
#'''Instead of plotting solute distribution as a % of the total, plot it as
# mols (no conversion to %)'''
#fig, ax = plt.subplots(figsize=(12,9), dpi=800)
#plt.plot(x, chan.iloc[:,1], label = 'Channel' ,linewidth = 3)
#plt.plot(x, frac.iloc[:,1], label = 'Fracture', linewidth = 3)
#plt.title('6mL/min Solute Distribution', fontsize=20)
#plt.xlabel('Time [s]', fontsize=18)
#plt.ylabel('Tracer [mol]', fontsize=18)
#plt.tick_params(axis='x', which='major', labelsize=16) # x-axis tick size
#plt.tick_params(axis='y', which='major', labelsize=16) # y-axis tick size
#plt.legend(loc = 'upper right', fontsize=16)
#plt.semilogy()
# %%
#Regenerating all of these plots but using mass (or amount, mols) instead of % of tracer or % discharge
#6mL/min
#Tracer distribution throughout the domain
savefig = False
florate = '009'
frac = pd.read_csv('fracture_C_009mL_integration_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
chan = pd.read_csv('Channel_C_009mL_integration_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])
inlet = pd.read_csv('inlet_block_C_009mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Concentration (mol)'])

#fix the weird issue with the diffusion out from the initial concentration box
frac.iloc[0,1] = 0
chan.iloc[0,1] = 0
x = frac.iloc[:,0]
fig, ax = plt.subplots(figsize=(12,9), dpi=800)
plt.plot(x, chan.iloc[:,1], label = 'Channel')
plt.plot(x, frac.iloc[:,1], label = 'Fracture')
ax.semilogy()
plt.title(f'{florate}mL/min Tracer Distribution, Semilog', fontsize=24)
plt.tick_params(axis='x', which='major', labelsize=16) # x-axis tick size
plt.tick_params(axis='y', which='major', labelsize=16) # y-axis tick size
plt.ylabel('Amount of Tracer [mol]', fontsize=18)
plt.xlabel('Time [s]', fontsize=18)
plt.legend(loc='upper right', fontsize=16)
if savefig:
    datadir = os.getcwd()
    savedir = 'D:\COMSOL_Files\COMSOL_BTC_Plots'
    os.chdir(savedir)
    plt.savefig(f'{florate}mL_min_solute_dist_mols_semilog')
    os.chdir(datadir) 

#%%
#The discharge for both simulations are constant throuhgout, so integrate and
#get the % discharge from the fracture and the channel out of the total discharge

#6mL/min
#import the datafiles
dtot =  pd.read_csv('total_discharge_6mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Velocity magnitude (m^3/s)'])
dfrac = pd.read_csv('fracture_discharge_6mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Velocity magnitude (m^3/s)'])
dchan = pd.read_csv('channel_discharge_6mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Velocity magnitude (m^3/s)'])
#calculate the total discharge across the entire outlet
dis_tot = np.trapz(dtot.iloc[:,1])
#calculate the discharge from the fracture only
dis_frac = np.trapz(dfrac.iloc[:,1])
#calculate the discharge from the channel only
dis_chan = np.trapz(dchan.iloc[:,1])

#fracture discharge %
frac_per = (dis_frac/dis_tot)*100
chan_per = (dis_chan/dis_tot)*100
#print the data
print(f'6mL/min fracture contribution to discharge: {frac_per}')
print(f'6mL/min channel contribution to discharge: {chan_per}')


#0.09mL/min
dtot =  pd.read_csv('total_discharge_009mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Velocity magnitude (m^3/s)'])
dfrac = pd.read_csv('fracture_discharge_009mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Velocity magnitude (m^3/s)'])
dchan = pd.read_csv('channel_discharge_009mL_V4.txt', skiprows=5, delim_whitespace=True, names=['Time', 'Velocity magnitude (m^3/s)'])
#calculate the total discharge across the entire outlet
dis_tot = np.trapz(dtot.iloc[:,1])
#calculate the discharge from the fracture only
dis_frac = np.trapz(dfrac.iloc[:,1])
#calculate the discharge from the channel only
dis_chan = np.trapz(dchan.iloc[:,1])

#fracture discharge %
frac_per = (dis_frac/dis_tot)*100
chan_per = (dis_chan/dis_tot)*100
#print the data
print(f'0.09mL/min fracture contribution to discharge: {frac_per}')
print(f'0.09mL/min channel contribution to discharge: {chan_per}')

