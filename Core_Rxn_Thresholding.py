# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 07:22:12 2023

@author: Charlie
"""

"""This is a code designed to take images of cores that have undergone
core flooding experiments and convert the images to greyscale. Then, these
greyscale images are thresholded to highlight the locations of mineral alteration
assuming that this alteration has a different color at the surface"""

#import the necessary package dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio.v3 as iio
import skimage.color
import skimage.filters
from PIL import Image
from skimage.measure import label, regionprops

# %%
#Change the directory to the directory with the images
path2data = "G:/.shortcut-targets-by-id/1dgvo5l64vNS4MWXXBf-luf9_WsUtbMnL/MN Carbonate Kang Collab/Core photos post experiment/Core Post Experiments/Thresholding_Images"

os.chdir(path2data)

#loop thru to put the image files into a python list
jpg_files = glob.glob(os.path.join(path2data, '*.jpg'))
im_list = [] #list of each of the jpg images to be thresholded

for jpg in jpg_files:
    img = Image.open(jpg)
    im_list.append(img)
# get the names of each of the files in order and associate 
# them with the images in the im_list list
fname_lst = []
for i in range(len(jpg_files)):
    file = jpg_files[i]
    w_ext = file.split('\\')[-1]
    word = w_ext.split('.')[0]
    fname_lst.append(word)
    
# merge these two lists together
imgs = list(zip(fname_lst, im_list))
# %%
#creating a greyscale image does not work well at all. It seems that, under all lighting conditions, the yellows/whites just do not show up well in the greyscale.
#hence, the greyscale has been replaced by an algorithm to recognize and plot the altered areas based on their yellowish color

#algorithm to create yellow color thresholds of the high light images only, save these plots as images
hlst = []
for f in range(len(imgs)):
    pic_name = imgs[f][0]
    if pic_name[3:7] == 'high':
        hlst.append((imgs[f][0],imgs[f][1]))

#crop the background out of each of the images before they are passed to 
#have their yellow regions isolated


#isolate the yellow alteration areas as shown on the surface of the rocks
for pic in range(len(hlst)):
    impath = os.path.join(path2data, hlst[pic][0]+'.jpg')
    image = iio.imread(impath)
    
   #crop the background out of each of the images before they are passed to 
   #have their yellow regions isolated 
    gray_image = skimage.color.rgb2gray(image)
    
    #threshold the grayscale image in order to obtain a binary mask
    
    t = 0.45 #adjust this threshold value to suit the coloring of your specific greyscale
    
    bin_mask = gray_image > t
    label_image = label(bin_mask)
    
    props = regionprops(label_image)
    largest_region = max(props, key = lambda r: r.area)
    
    #extract te bounding box coords of the largest regions
    y_min, x_min, y_max, x_max = largest_region.bbox
    
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    #highlight the yellow areas expressing alteration within the image itself
    hsv_image = skimage.color.rgb2hsv(cropped_image)
    lower_yellow = np.array([0.05, 0.23, 0.5])   # Hue, Saturation, Value
    upper_yellow = np.array([0.23, 1.0, 1.0]) 
    
    mask = np.logical_and(np.all(hsv_image >= lower_yellow, axis=-1), np.all(hsv_image <= upper_yellow, axis=-1))
    
    yellow_off_yellow_regions = np.zeros_like(cropped_image)
    yellow_off_yellow_regions[mask] = cropped_image[mask] 

    plt.imshow(yellow_off_yellow_regions)
    plt.axis('off')  # Optional: Turn off axis ticks and labels
    plt.savefig('thresh_crop' + hlst[pic][0]+'.jpg')
    plt.show()
# %%
'''The code section above uses yellow values on the cores in order to establish 
and outline the regions that have been altered. I would like to try again using
a thresholding algorithm that uses a greyscale as I worry that the yellow 
method may in fact be too subjective? Update. greyscaling is really hard because
I am trying to select for a certain color of light'''
#now attempting using adaptive thresholding based on skimage instead of global thresholding using PILLOW alone
# %%
#for each of the images that have been cropped,
#calculate the % of the surface that shows mineral alteration (by color)
ylst = []
for f in range(len(imgs)):
    pic_name = imgs[f][0]
    if pic_name[0:6] == 'thresh':
        ylst.append((imgs[f][0]))

per_altered = []
for i in range(len(ylst)):
    impath = os.path.join(path2data, ylst[i]+'.jpg')
    image = iio.imread(impath)
    #get the height, width, # of channels for the image being considered
    height, width, chan = image.shape

    #calculate the total number of pixes by multiplying width * height

    total_pixels = width * height
    
    #calculate the total number of non-black pixels by using a greyscale
    grayscale = skimage.color.rgb2gray(image)

    colpix = len(grayscale[grayscale > 0])
    
    #calculate the percentage of the image that is non-black pixels (altered region)
    alt_percent = colpix / total_pixels
    
    per_altered.append((ylst[i], alt_percent * 100))
    
savefile = os.path.join(path2data, 'Alteration_percent.txt')

with open(savefile, 'w') as file:
    content = '\n'.join(map(str, per_altered))
    # Write the content to the file
    file.write(content)