# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 08:36:18 2023

@author: Charlie
"""
import numpy as np
from matplotlib import pyplot as plt
import os
# %%
os.chdir('C:/Users/Charlie/Desktop')
L = np.arange(0,25) #length in mm
D = 6.7e-4 #mm^2/s

tarr = []

for i in range(len(L)):
    tarr.append(((L[i] **2)/D)/3600) #tarr in hours

plt.plot(L,tarr)
plt.grid(True)
plt.title('Diffusion timescales (D=6.4e-7[mm^2/s])')
plt.xlabel('Length L [mm]')
plt.ylabel('time (hours)')
plt.savefig('tdiff.png')