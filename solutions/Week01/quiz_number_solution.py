#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:17:42 2021

@author: abda
"""

import os
import numpy as np
import scipy.ndimage
import skimage.io
import matplotlib.pyplot as plt


#%% Read in image data
file_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(file_path, '..', 'Data', 'week1', 'noisy_number.png')

I_noisy = skimage.io.imread(path).astype(np.float)
sigma = 15;
I_smoothed = scipy.ndimage.gaussian_filter(I_noisy, sigma, mode='nearest')

fig, ax = plt.subplots()
ax.imshow(I_smoothed)
plt.show()
