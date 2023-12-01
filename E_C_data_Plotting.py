# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:23:20 2023

@author: csouc
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
# %%
# set the current working directory
path = 'C:/Campbellsci/PC400'

os.chdir(path)

#list the names of the files in the directory
os.listdir()

#select the name of the file that you want to be read into the plotter
filename = 'Bear_Spring_EC_TEMP_UMN_Table1.dat'

def reader(filename):
    with open(filename, "r") as my_file:
        Contents = my_file.read()
    return Contents

# %% call the reading function
# Assuming the provided data is stored in a file named 'data.dat'
# You might need to adjust the delimiter based on the actual format of your .dat file
data = pd.read_csv(filename, delimiter=',', skiprows=1)

# Extract the desired columns
selected_columns = ['TIMESTAMP', 'Cond_Avg', 'Ct_Avg', 'Temp_C_Avg']
selected_data = data[selected_columns]
selected_data = selected_data.dropna()

# Print the selected data
print(selected_data)

#%%
# plot the selected data

x = selected_data['TIMESTAMP']


y = selected_data['Cond_Avg']

x = x[2:]

days = np.arange(0, len(x)) * 15 * 0.000694444

y = y[2:]
y = y.dropna()
y = y.to_numpy()


y_float = []
for i in range(len(y)):
    y_float.append(float(y[i]))
    
# Plot the data
fig = plt.figure()

# Using a scatter plot with marker="o" and specifying the tick values
plt.scatter(days, y_float, marker="o")

# Set y tick values and labels
plt.yticks([0, 0.5, 1, 1.5, 2], ['0', '0.5', '1', '1.5', '2'])
plt.ylabel('Conductivity (T corrected) [mS/cm]')
plt.xlabel('Elapsed Days')
plt.title('Bear Spring Conductivity, May to August 2023')
# Other plot customizations...
#plt.savefig('May_to_August_Conductivity_Bear_Spring.png', dpi=800)
plt.show()







