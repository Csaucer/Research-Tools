# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:23:20 2023

@author: csouc
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as DT # for use in the weird CR510 file structure data
from datetime import timedelta
from datetime import time as Tmod
import os
# %%
# set the current working directory
path = 'D:/Bear_Spring_Field_Data'

os.chdir(path)

#list the names of the files in the directory
os.listdir()

#generate a reader for reading the .dat file
def reader(filename):
    with open(filename, "r") as my_file:
        Contents = my_file.read()
    return Contents


#set up a generalized plotting tool based on datetime objects
def plot_data(df, x_column, y_column, T_column, title='', xlabel='', ylabel='', marker='', EC_color='',
               T_color='', titlesize = 15, fontsize=12, savefig=False, save_title=''):
    fig, ax = plt.subplots(figsize=(12, 9))# create a new plot for the EC, temp, precipitation data
    ax.plot(df[x_column], df[y_column], marker=marker, color=EC_color, label=y_column)#plot the EC data (first plotted object)
    ax.set_title(title, fontsize=titlesize) #set the plot title
    ax.set_xlabel(xlabel, fontsize=fontsize) #set the xlabel(DateTime)
    ax.set_ylabel(ylabel, fontsize=fontsize) #Set the label on the y-axis for the EC plot
    ax.tick_params(axis='x', rotation=45, labelsize=fontsize) #rotate the labels on the x-axis so that they can fit on the graph
    ax.tick_params(axis='y', labelcolor=EC_color, labelsize=fontsize) #alter the color, size of the tick marks
    #ceate a secondary y-axis on the right side of the plot for the temperature data using twinx
    ax2 = ax.twinx()
    ax2.plot(df[x_column],df[T_column], marker=marker, color=T_color, label=T_column) #plot the temp data
    ax2.set_ylabel(T_column, color=T_color, fontsize=fontsize) #set the label to the correct color
    ax2.tick_params(axis='y', labelcolor=T_color, labelsize=fontsize) #set the tick marks to the correct color
    ax2.spines['right'].set_color(T_color) #set the spine of the plot (the right boundary of the box) to the right color
    if savefig:
        os.chdir('D:/Bear_Spring_Field_Data/Field_data_plots')
        plt.savefig(f'{save_title}.png', dpi=800)
        os.chdir('D:/Bear_Spring_Field_Data')
    plt.show() #show the plot


# %% call the reading function
# Assuming the provided data is stored in a file named 'data.dat'
# You might need to adjust the delimiter based on the actual format of your .dat file
#select the name of the file that you want to be read into the plotter
filename = 'Bear_Spring_EC_TEMP_UMN_Table1.dat'

data = pd.read_csv(filename, delimiter=',', skiprows=1)

# Extract the desired columns
selected_columns = ['TIMESTAMP', 'Cond_Avg', 'Ct_Avg', 'Temp_C_Avg']
data = data[selected_columns]
data = data.dropna()

#drop the first row that contains the units (if you need to check units put a cell break before this row)
data.drop(data.index[0:3],axis=0, inplace=True)
#convert the date and time from this probe to a datetime object for use
#in the plot_data() function
#from the CR800 dataloggers the timestamp data is stored as a string and should be easy to conver to Datetime objects
for i, row in data.iterrows():
    data.at[i, 'datetime'] = DT.strptime(row['TIMESTAMP'], '%Y-%m-%d %H:%M:%S')
    
#the CR800 saves values as strings by default. Need to convert them to float32
data['Cond_Avg'] = data['Cond_Avg'].astype(np.float32)
data['Ct_Avg'] = data['Ct_Avg'].astype(np.float32)
data['Temp_C_Avg'] = data['Temp_C_Avg'].astype(np.float32)



# plot the data
plot_data(data, 'datetime', 'Cond_Avg', 'Temp_C_Avg' , title='Bear Spring EC Dec. 2023-Jan. 2024',
          xlabel='Date', ylabel='Electrical Conductivity [mS/cm]', marker='.', EC_color='k', T_color ='r', savefig=False, save_title='BS_Dec2023_Jan2024')#minimum of 0.005 mS/cm
# %%

#construct a system capable of interpreting the data collected at Hammel spring
#where the datalogger box got knocked over and covered with snow
#the file structure on this (older datalogger CR510) is all messed up
#select the name of the file that you want to be read into the plotter
filename = 'Hammel_maybe_broken_1_23_24.dat'
#convert to a pandas dataframe and cut the rows off that are not representative
# of the actual time post install
data = pd.read_csv(filename, delimiter=',', skiprows=17)

#trim off all the unecessary columns that are created for no reason
data = data.drop(data.columns[0], axis=1)

#change the names of the colums to be interpretable names
new_names = {'2023': 'yr', '353':'day', '1145': 'time', '.287': 'Cond' ,'-.85943': 'Ct', '5.49': 'TempC', '12.83': 'BattV'}
data = data.rename(columns=new_names)

#construct an algorithm capable of reading the messed up file structure
#this function will combine the first three columns into an appropriate 
#datetime collumn

def datefix(broken_df):
    new_col = 'datetime' #create a new column that will become the datetime for the future plots
    new_val = pd.Series([None] * len(broken_df)) #create an empty value set to fill the new column
    broken_df.insert(3, new_col, new_val)
    #for each of the rows in the new df, calculate and add a new datetime object
    for i in range(len(data)):
        yr = data.yr[i] #get the year for the current row
        day = data.day[i] #get the day(cumulative value) for the row
        day = int(day) #convert this value to a regular python integer
        time = data.time[i] #get the time value in military time e.g. 1000 
        if time == 2400:
            time = 0  # Convert 2400 to 0000 (midnight)
            day = day + 1
        start_date = DT(yr,1,1) #set the start date for the counting module
        target_date = start_date + timedelta(days = day-1)
        hour = time//100
        minute = time % 100
        tval = Tmod(hour, minute)
        newDT = target_date.replace(hour=tval.hour, minute = tval.minute)
        #insert the new datetime object into the appropriate column and row
        data.at[i, 'datetime'] = newDT
    return broken_df

data = datefix(data)

#After working on this tool to get everything into the appropriate datetime format
#I think this is really the eaisest way to plot graphs in a clear, consise way.
# moving forward with timeseries data collected from field work, I will plan
# to convert everything into a datetime object to make plotting easier


#Plot the Hammel data from Dec.2023-Jan.2024
plot_data(data, 'datetime', 'Cond', 'TempC' , title='Hammel Spring EC Dec. 2023-Jan. 2024',
          xlabel='Date', ylabel='Electrical Conductivity [mS/cm]', marker='.', EC_color='k', T_color ='r', savefig=False, save_title='Hammel_Dec2023_Jan2024')#minimum of 0.005 mS/cm

# %% plot the Data from Bear Creek
filename = 'BC_Downstream_BS_1_23_24.csv'
data = pd.read_csv(filename, delimiter=',', skiprows=1)
#trim off the unecessary rows
data.drop(data.tail(3).index, inplace=True)
#trim off the unecessary columns
data.drop(data.columns[5:], axis=1, inplace=True)
data.drop(data.columns[0], axis=1, inplace=True)
#make the column headers more readable
in_name = data.columns.values
data = data.rename(columns={in_name[0]:'Timestamp' ,in_name[1]: 'Cond_lowrange',
                     in_name[2]: 'Cond_highrange', in_name[3]: 'TempF'})
#convert the temperature values from F to C and rename the column to 'TempC'
data['TempF'] = (data['TempF'] - 32) * (5/9)
#convert the conductivity data from uS/cm to mS/cm to match all the campbell probe data
data['Cond_lowrange'] = (data['Cond_lowrange']/1000)
data['Cond_highrange'] = (data['Cond_highrange']/1000)
data.rename(columns={'TempF': 'TempC'}, inplace=True)
#create a datetime column from the Timestamp data by converting to a datetime object
for i, row in data.iterrows():
    data.at[i, 'datetime'] = DT.strptime(row['Timestamp'], '%m/%d/%y %I:%M:%S %p')

#Plot the BC data from Dec.2023-Jan.2024
plot_data(data, 'datetime', 'Cond_highrange', 'TempC' , title='Bear Creek Downstream From Bear Spring EC Dec. 2023-Jan. 2024',
          xlabel='Date', ylabel='Electrical Conductivity [mS/cm]', marker='.', EC_color='k', T_color ='r', savefig=False, save_title='BC_downstream_of_BS_Dec2023_Jan2024')
# %%
# %% plot the Data from Bear Creek
filename = '55A405_Spring_1_23_24.csv'
data = pd.read_csv(filename, delimiter=',', skiprows=1)
#trim off the unecessary rows
data.drop(data.tail(3).index, inplace=True)
#trim off the unecessary columns
data.drop(data.columns[5:], axis=1, inplace=True)
data.drop(data.columns[0], axis=1, inplace=True)
#make the column headers more readable
in_name = data.columns.values
data = data.rename(columns={in_name[0]:'Timestamp' ,in_name[1]: 'Cond_lowrange',
                     in_name[2]: 'Cond_highrange', in_name[3]: 'TempF'})
#convert the temperature values from F to C and rename the column to 'TempC'
data['TempF'] = (data['TempF'] - 32) * (5/9)
#convert the conductivity data from uS/cm to mS/cm to match all the campbell probe data
data['Cond_lowrange'] = (data['Cond_lowrange']/1000)
data['Cond_highrange'] = (data['Cond_highrange']/1000)
data.rename(columns={'TempF': 'TempC'}, inplace=True)
#create a datetime column from the Timestamp data by converting to a datetime object
for i, row in data.iterrows():
    data.at[i, 'datetime'] = DT.strptime(row['Timestamp'], '%m/%d/%y %I:%M:%S %p')

#Plot the spring 55A405 data from Dec.2023-Jan.2024
plot_data(data, 'datetime', 'Cond_highrange', 'TempC' , title='Spring 55A405EC Dec. 2023-Jan. 2024',
          xlabel='Date', ylabel='Electrical Conductivity [mS/cm]', marker='.', EC_color='k', T_color ='r', savefig=False,save_title='Spring_55A405_Dec2023_Jan2024')