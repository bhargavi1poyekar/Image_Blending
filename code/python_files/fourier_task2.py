import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Reference: https://stanford.edu/class/ee367/reading/OlivaTorralb_Hybrid_Siggraph06.pdf

A hybrid image (H) is obtained by combining two images (I1 and
I2), one filtered with a low-pass filter (G1) and the second one filtered with a high-pass filter 
(1 − G2): H = I1 · G1 + I2 ·(1 − G2), the operations are defined in the Fourier domain. Hybrid images
are defined by two parameters: the frequency cut of the low resolution image (the one to be seen at a far distance), 
and the frequency cut of the high resolution image (the one to be seen up close). An additional parameter can be 
added by introducing a different gain for each frequency channel. For the hybrids shown in this paper we
have set the gain to 1 for both spatial channels. We use gaussian filters (G1 and G2) for the low-pass 
and the high-pass filters. We define the cut-off frequency of each filter as the frequency for with
the amplitude gain of the filter is 1/2.

'''

# Read the images
low_freq_image = cv2.imread("../input_images/cat.jpg") # I want the lower frequencies of cat
high_freq_image = cv2.imread("../input_images/panda.jpg") # I want high frequencies of panda

# Apply Low pass filter to cat 
lf_blurred = cv2.GaussianBlur(low_freq_image, (11, 11), 8)
# (11,11)=> Kernel Size, I tried various sizes to see the different hybrids. With lower kernel sizes,
# The smoothening was not that good. With higher kernel size, I got more smoothening effect.

# 8 => stands for standard deviation (sigma). # greater the sigma=> Averages pixel values over wider area. 

# Apply high pass filter to panda 
hf_blurred = cv2.GaussianBlur(high_freq_image, (11, 11), 2)

# We get high pass filter by subtracting the low pass filtered version from original image
hf_filtered = high_freq_image - hf_blurred

# Combine the images
hybrid_image = lf_blurred + hf_filtered

# Ssave the hybrid image
cv2.imwrite("../output_images/panda_cat_hybrid.png", hybrid_image)

# Display original and hybrid images
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(cv2.cvtColor(low_freq_image, cv2.COLOR_BGR2RGB))
plt.title("Cat Image")

plt.subplot(132)
plt.imshow(cv2.cvtColor(high_freq_image, cv2.COLOR_BGR2RGB))
plt.title("Panda Image")

plt.subplot(133)
plt.imshow(cv2.cvtColor(hybrid_image, cv2.COLOR_BGR2RGB))
plt.title("Hybrid Image")

plt.tight_layout()
plt.show()