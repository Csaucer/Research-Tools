# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:21:40 2023

@author: Charlie
"""

'''General script for taking a series of time series images from the same 
orientation and animate them into a short movie. Should work with all
images in this form, designed to work with .png but .jpg, etc. should also work'''

#load the necessary package dependencies
import os


# %%
#trying a different (and likely better) method that will allow for frame duration control
from PIL import Image
#animating the plots in order to make data more viewable as a time series
#modified from the code above to create animated GIFs
images = []
time_steps = 7
image_dir = r'G:/My Drive/Core Flooding Project/PET_RAW_Finalized/pre_reaction/1C_Pre_3D/009ml_min_mayavi'

os.chdir(image_dir)
#load all of the images into the images list
# Loop through the directory and add PNG files to the list
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        images.append(os.path.join(image_dir, filename))

# Create a GIF from the saved plot images
frames = []
for image in images:
    frames.append(Image.open(image))

# Save the frames as an animated GIF
frames[0].save('output_animation.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=1000,  # Set the delay between frames (in milliseconds)
               loop=0)  # Set loop to 0 for an infinite loop, or any other positive integer for a finite loop
