# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:59:58 2023

@author: csouc
"""
"""Developing automatic plotting code to plot stable isotope data.
Capable of being run via the command prompt"""

"""Like most python based tools, the inputs that this tool takes are very
specific to the desired outputs. This code has been designed with the goal
of creating a tool that can be used to easily update plots of new isotopic
data being output from the  'Picarro' MATLAB tool created by Seonkyoo Yoon
at UMN for processing and correcting stable isotope data produced by the 
picarro cavity ringdown spectrometer."""

"""Because this code was specfiically built for the data structure of stable
isotope data from Bear Spring and its surrounding region, it has many funcitnos
and formats that are not going to be useful for other isotope applications.
As such, it is necssary to edit this code in order to improve the useability
for your specific isotope datasets. There are comments throughtout the code
that explain what things do and how they may need to be changed to suit your
own purposes."""

"""The file structure used here has no naming convention for the .csv files
input initially but it does have a naming convention for each of the samples.
That name structure is as follows: XX_---_YYMMDD, where the XX is the intiial 2 or 3
letter identifier of the locaiton, the --- represents any additional location info,
and the YYMMDD represents the date value. It is imperative that this structure of the 
data is followed in order to get the proper results """


"""Import package dependencies"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from itertools import groupby
from collections import defaultdict
#Set the appropriate working directory, may need to change if on different machine
os.chdir(r'C:\Users\csouc\Isotopes\Bear_Spring') 


#%%

"""define functions for model"""
#Scrape through the directory and get file names, store in a matrix
def get_file_names(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names


#Read through the files and load in all of the data to python
def get_file_data(file_names):
    file_data = []
    for filename in file_names:
        suff = ".csv"
        res = list(filter(filename.endswith, suff)) !=[]
        if res == True:
            file_data.append(pd.read_csv(filename))
    return file_data

#Scan through a pandas dataframe and find the row of a keyword ("Samples")
#this exculdes all of the calibration data seen at the top of the .csv
def find_keyword_index(dfs, keyword):
    keyword_index = []
    for df in dfs:
    # Iterate over each row of the first column of the DataFrame
        for index, row in df.iterrows():
            if row.str.contains(keyword, case=False).any():
                keyword_index.append(index)
    # Return the index of the first matching row
    return keyword_index
    
    # If the keyword is not found
    return -1

def make_row_indexes(keyword_indexes):
    row_indexes = []
    for index in keyword_indexes:
        row_indexes.append(np.arange(0,index+1))
    return row_indexes

#Trim the dataset down to only the id, d18_O, sigma_18_O, dD, sigma_dD
#remove all of the unecessary rows that don't include the sample data   
def trim_datasets(dataframes, row_indexes, column_indexes):
    file_data_short = dataframes.copy()
    for i,df in enumerate(file_data_short):
            #remove columns based on indexes
            df.drop(df.index[row_indexes[i]], axis=0,\
                                        inplace=True)
    
            #remove columns based on indexes
            df.drop(df.columns[column_indexes], axis=1,\
                                        inplace=True)
    file_data = get_file_data(file_names)
#though this re-initialization of file_data is inefficient, it prevents an error occuring
#when the code is re-run after the first pass, as pandas cannot separate
#copied variables (no deep copy support, file_data_short is always connected
#to dataframes)
    return file_data_short, file_data

def make_header(trimmed_dataset):
    data_header = []
    for df in trimmed_dataset:
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        data_header.append(df)
    return(data_header)

def reset_index(dataframes):
    for df in dataframes:
        # Reset the index of the DataFrame starting from 0
        df.reset_index(drop=True, inplace=True)

##should probably run this before the make_header to pull spaces out from header
            
def remove_spaces(dataframes, column_name):
    for df in dataframes:
        # Apply lambda function to remove spaces from each cell in the specified column
        df[column_name] = df[column_name].apply(lambda x: x.replace(" ", ""))
        #Apply the lamdba function to remove spaces from each cell in 1st row
        df.iloc[0] = df.iloc[0].str.strip()
    
    
#need to alter the names in DF index 1 in order to have location reference to 
#bear spring
    
#isotopes will be put into lists of tuples from the dataframes based on their
#location and their date
def separate_isotopes(dataframes):
    BS = []  #each of these lists is specific for the data at Bear Spring.
    BC = [] #to implement for your own purposes, these can be changed to group
    Snow = [] #your data however you feel it is necessary, just be sure to change
    other = [] #the conditions in the function below
    for df in dataframes:
        for i in range(len(df)):
            string = df.iloc[i,0]
            if string[:2] == 'BS':
                name = ''.join([i for i in string if not i.isdigit()])
                n = 6
                date = string[-n:]
                d18O = df.iloc[i,1]
                dD =  df.iloc[i,3]
                sig18O = df.iloc[i,2]
                sigD = df.iloc[i,4]
                BS.append((name, date, d18O, dD, sig18O, sigD))
            elif string[:2] == 'BC':
                name = ''.join([i for i in string if not i.isdigit()])
                n = 6
                date = string[-n:]
                date = string[-n:]
                d18O = df.iloc[i,1]
                dD =  df.iloc[i,3]
                sig18O = df.iloc[i,2]
                sigD = df.iloc[i,4]
                BC.append((name,date,d18O, dD, sig18O, sigD))
            elif string[:2] == 'Sn':
                name = ''.join([i for i in string if not i.isdigit()])
                n = 6
                date= string[-n:]
                date = string[-n:]
                d18O = df.iloc[i,1]
                dD =  df.iloc[i,3]
                sig18O = df.iloc[i,2]
                sigD = df.iloc[i,4]
                Snow.append((name,date, d18O, dD, sig18O, sigD))
            else:
                name = ''.join([i for i in string if not i.isdigit()])
                n = 6
                date = string[-n:]
                date = string[-n:]
                d18O = df.iloc[i,1]
                dD =  df.iloc[i,3]
                sig18O = df.iloc[i,2]
                sigD = df.iloc[i,4]
                other.append((name,date, d18O, dD, sig18O, sigD))
    return BS, BC, Snow, other
        

"""The plotting tools all take a dataframe as the input, as well as a string
that is the same as that dataframe name (e.g. plot_by_date(BS, 'BS') """
#this plots all of the data by dates, without any consideration of location
def plot_by_date(dataframe, dataframe_name):
    #sort the data by dates
    sorted_data = sorted(dataframe, key=lambda x: x[1])
    #group the data by the dates
    grouped_data = groupby(sorted_data, key=lambda x: x[1])
    
    #prepare lists for plotting
    dD_values = []
    d18O_values = []
    #prep list for labels
    labels = []
    #iterate over the grouped data
    for date, group in grouped_data:
        group_dD = []
        group_d18O = []
        for _, _, dD, d18O, _, _ in group:
            try:
               dD_float = float(dD)
               d18O_float = float(d18O)
               group_dD.append(dD_float)
               group_d18O.append(d18O_float)
            except ValueError:
               pass
        dD_values.append(group_dD)
        d18O_values.append(group_d18O)
        
        labels.append(date)  # Add date as a label
    #plot the dD against the d18O for each group
    for dD, d18O in zip(dD_values, d18O_values):
        plt.scatter(dD, d18O)
    #customize the plot
    plt.xlabel('dD')
    plt.ylabel('d18O')
    
    # Adjust the x-axis with increased number of tick marks
    x_ticks = np.linspace(min(min(dD_values)), max(max(dD_values)), num=10)
    plt.xticks(x_ticks)
    
    # Adjust the y-axis with uniform step size
    y_ticks = list(range(int(min(min(d18O_values))), int(max(max(d18O_values))) + 1, 1))
    plt.yticks(y_ticks)
    plt.legend(labels)
    plt.gca().xaxis.tick_top()
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.gca().xaxis.set_label_position('top')
    plt.title(f'Isotopes by date for {dataframe_name}')  # Add variable name to the plot title
    plt.savefig(f'Isotopes_by_date_{dataframe_name}.png', dpi=600)

    plt.show()
            
# THis plots all of the isotope data by location only with no consideration
#of the dates
def plot_by_location(dataframe, dataframe_name):
    sorted_data = sorted(dataframe, key =lambda x: x[0])
    #group the data by the dates
    grouped_data = groupby(sorted_data, key=lambda x: x[0])
    
    #prepare lists for plotting
    dD_values = []
    d18O_values = []
    #prep list for labels
    labels = []
    #iterate over the grouped data
    for date, group in grouped_data:
        group_dD = []
        group_d18O = []
        for _, _, dD, d18O, _, _ in group:
            try:
               dD_float = float(dD)
               d18O_float = float(d18O)
               group_dD.append(dD_float)
               group_d18O.append(d18O_float)
            except ValueError:
               pass
        dD_values.append(group_dD)
        d18O_values.append(group_d18O)
        
        labels.append(date)  # Add date as a label
    #plot the dD against the d18O for each group
    for dD, d18O in zip(dD_values, d18O_values):
        plt.scatter(dD, d18O)
    #customize the plot
    plt.xlabel('dD')
    plt.ylabel('d18O')
    
    # Adjust the x-axis with increased number of tick marks
    x_ticks = np.linspace(min(min(dD_values)), max(max(dD_values)), num=10)
    plt.xticks(x_ticks)
    
    # Adjust the y-axis with uniform step size
    y_ticks = list(range(int(min(min(d18O_values))), int(max(max(d18O_values))) + 1, 1))
    plt.yticks(y_ticks)
    plt.legend(labels)
    plt.gca().xaxis.tick_top()
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.gca().xaxis.set_label_position('top')
    plt.title(f'Isotopes by location for {dataframe_name}')  # Add variable name to the plot title
    plt.savefig(f'Isotopes_by_location_{dataframe_name}.png', dpi=600)

    plt.show()
            
#This plots all of the isotope data by both location and by date.
def plot_isotope_data(lists):
    # Prepare lists for plotting
    grouped_data = defaultdict(list)

    for list_name, tuples_list in lists:
        for tuple_data in tuples_list:
            dD = float(tuple_data[2])
            d18O = float(tuple_data[3])
            date = tuple_data[1]
            grouped_data[(list_name, date)].append((dD, d18O))

    # Plot the data for each group
    for group, data in grouped_data.items():
        list_name, date = group
        dD_values, d18O_values = zip(*data)
        plt.scatter(dD_values, d18O_values, label=f'{list_name} - {date}')

    # Customize the plot
    plt.xlabel('dD')
    plt.ylabel('d18O')
    plt.title('Location and Date Isotope Data Plot')
    plt.gca().xaxis.tick_top()
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.gca().xaxis.set_label_position('top')
    plt.savefig('Isotopes_by_location_and_date.png', dpi=600)
    plt.legend()
    plt.show()

    
# %%
'''get csv data from directory into python. Be sure this code is in the appropriate
directory to access the data'''
directory_path = os.getcwd()
file_names = get_file_names(directory_path)
file_data = get_file_data(file_names)

#%%
keyword = "Samples"
keyword_indexes= find_keyword_index(file_data, keyword)
#%%
'''Extract the useful data out from the file_data dataframe'''
##search through to find the index of the keyword "Samples" which is included
#in the csv files one column above the column headings for the isotope samples
column_indexes = [1,2,3,4,9] #fixed columns that will be removed from all dataframes
row_indexes = make_row_indexes(keyword_indexes)
trimmed_dataset, file_data = trim_datasets(file_data, row_indexes, column_indexes)

#%%
'''clean up these trimmed datasets to have headers, no spaces in names'''
#reset the indexes for the datasets with extra information cut out
reset_index(trimmed_dataset)
#create a list of all of the headers of the trimmed_datasets with their 
#inexes reset. Note that this is BEFORE the final, official column headers
#have been created
column_headers = list(trimmed_dataset[0].columns.values)
#access the first column_header and save it for use in the remove spaces fxn
column_name = column_headers[0]
#remove the spaces from the values in the first column and first row
remove_spaces(trimmed_dataset, column_name)
#move the first row into the headers
iso_datasets = make_header(trimmed_dataset)
#%%
"""Set up conventions for plotting isotopes"""

#import LMWL and GMWL data
#because there is not any standardization with the LMWL and GMWL datasets,
#these will need to be set up for import by hand depending upon where the data
#is coming from. If these datsets have been provided by Charlie Soucey from
#the data collected by Scott Alexander at University of Minnesota TC, then
#no additional edits need to be made beyond making sure the datasets are in the
#correct directory

GMWL = pd.read_csv("MWLs/GNIP_Global_data_csv.csv",header=2,encoding = "ISO-8859-1"\
                   ,usecols = ['D','18O'])
GMWL.dropna(inplace=True) # remove the cells with nan


LMWL = pd.read_csv('MWLs/Minneapolis_IsoWater_csv.csv', header=0,\
                   usecols=['d(D_H)Mean','d(18_16)Mean','d(D_H)_SD','d(18_16)_SD'])
LMWL.dropna(inplace=True) # remove all the cells with nan from LMWL

#%%
'''Plot LMWL and GMWL'''

fig1,ax1 = plt.subplots()
plt.scatter(GMWL['18O'],GMWL['D'], label = 'GMWL')
plt.scatter(LMWL['d(18_16)Mean'], LMWL['d(D_H)Mean'], label = 'MPLS_MWL')

#plot the LMWL with error bars
fig2,ax2 = plt.subplots()
plt.scatter(LMWL['d(18_16)Mean'], LMWL['d(D_H)Mean'])
plt.errorbar(LMWL['d(18_16)Mean'],LMWL['d(D_H)Mean'],xerr=LMWL['d(18_16)_SD'],\
             yerr = LMWL['d(D_H)_SD'],fmt="o")


# %%
#Separate all of the data out into their respective location groups with the
#associated dates and isotope values. THis is obviously data dependent
BS, BC, Snow, other = separate_isotopes(iso_datasets)
#%%
"""Call the plotting functions above in order to plot the data in the
desired formats"""


'''Plot the isotopes by date'''
plot_by_date(BS, 'BS')
plot_by_date(BC,'BC')
plot_by_date(Snow, 'Snow')
plot_by_date(other, 'Other')

'''plot by location'''
plot_by_location(BS, 'BS')
plot_by_location(BC, 'BC')
plot_by_location(Snow, 'Snow')
plot_by_location(other, 'Other')

'''Plot by both location and date'''
lists = [
    ('BS', BS),
    ('BC',BC),
    ('Snow', Snow),
    ('Other',other)
    ]
plot_isotope_data(lists)


