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
import skimage.color
import skimage.filters
from PIL import Image
import cv2 as cv
# %%
# Change the directory to the directory with the images
path2data = "G:/.shortcut-targets-by-id/1dgvo5l64vNS4MWXXBf-luf9_WsUtbMnL/MN Carbonate Kang Collab/Core photos post experiment/Core Post Experiments/Thresholding_Images"
os.chdir(path2data)

# Loop through to put the image files into a Python list
jpg_files = glob.glob(os.path.join(path2data, '*.jpg'))
imgs = []  # List to store tuples of (filename, image)

for jpg in jpg_files:
    # Open the image file and convert it to RGB format
    with Image.open(jpg) as img:
        img_rgb = img.convert('RGB')
        # Get the filename without extension
        fname = os.path.splitext(os.path.basename(jpg))[0]
        # Append filename and RGB image to the list
        imgs.append((fname, img_rgb))
# %%
#creating a greyscale image does not work well at all. It seems that, under all lighting conditions, the yellows/whites just do not show up well in the greyscale.
#hence, the greyscale has been replaced by an algorithm to recognize and plot the altered areas based on their yellowish color

#create greyscale images of the top half of core 1C and core 2C
for i in range(len(imgs)):
    word1 = 'top' #only include the top halves of the images to avoid shadow effects
    word2 = 'thresh' #exclude the arleady thresholded images
    if word1 in imgs[i][0] and word2 not in imgs[i][0]:
        name = imgs[i][0] #collect the filenames of each image
        image = imgs[i][1] # get the image and assign it to a variable
        grey_image = skimage.color.rgb2gray(image)
        #thresh = skimage.filters.threshold_otsu(grey_image) #threshold the greyscale image
        thresh = 0.25
        binary = grey_image  > thresh #create a binary image map
        
        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1)
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])
       
        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Original')
        ax[0].axis('off')
       
        ax[1].hist(grey_image.ravel(), bins=256)
        ax[1].set_title('Histogram')
        ax[1].axvline(thresh, color='r')
       
        ax[2].imshow(binary, cmap=plt.cm.gray)
        ax[2].set_title('Thresholded')
        ax[2].axis('off')
       
        plt.show()
# %%
#grey value simple thresholding doesn't work very well. Attempting with hsv

#create hsv images of the top half of core 1C and core 2C
for i in range(len(imgs)):
    word1 = 'top' #only include the top halves of the images to avoid shadow effects
    word2 = 'thresh' #exclude the arleady thresholded images
    word3 = 'trim'
    word4 = 'high' #the high light versions of the photos seem to be more accurate
    #if word1 in imgs[i][0] and word2 not in imgs[i][0]:
    if word3 in imgs[i][0] and word4 in imgs[i][0]:
        name = imgs[i][0] #collect the filenames of each image
        image = imgs[i][1] # get the image and assign it to a variable'
        print(name)
        cv_im = np.array(image)
        cv_im = cv_im[:, :, ::-1].copy()
        # Convert the image to HSV color space
        hsv_image = cv.cvtColor(cv_im, cv.COLOR_BGR2HSV)
        
        lower_yellow = np.array([15, 65, 75])
        upper_yellow = np.array([35, 255, 255])
        #Create a mask based on the specified range of colors (yellow)
        mask = cv.inRange(hsv_image, lower_yellow, upper_yellow)
       
        # Apply the mask to the original image
        result = cv.bitwise_and(cv_im, cv_im, mask=mask)
       
        # Convert the result back to PIL Image if needed
        result_image = Image.fromarray(result[:, :, ::-1])  # Convert BGR to RGB
        plt.imshow(result_image)
        altered_pixels = np.sum(mask != 0)  # Count the number of altered pixels
        total_pixels = mask.size  # Total number of pixels in the image

        # Calculate the percentage of altered pixels
        percentage_altered = (altered_pixels / total_pixels) * 100

        print(f"Percentage of altered pixels: {percentage_altered:.2f}")
        plt.axis('off')
        plt.show()