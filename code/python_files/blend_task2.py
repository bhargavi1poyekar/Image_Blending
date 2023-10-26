import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the img
original = cv2.imread("../input_images/elephant.jpeg")
img=original # Getting the original img in img, to convert it. I will use original img for display purposes. 

gaussian = [img] # Stores the different Gaussian Pyramid Levels
laplacian = [] # Stores the different Laplacian Pyramid Levels

# Create Gaussian Pyramid to as smallest level as possible. 
while img.shape[0] > 1 and img.shape[1] > 1: # shape[0]=>height, shape[1]=> width
    img = cv2.GaussianBlur(img, (11,11), 2) # Step 1 => Smoothen the img using Gaussian Blur
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)) # Downsample the img to half it's size. 
    gaussian.append(img) # Store this level

# Create Laplacian Pyramids
for i in range(len(gaussian) - 1): # For all the gaussian pyramid levels, except the first one (original img). 
    upsampled = cv2.resize(gaussian[i + 1], (gaussian[i].shape[1], gaussian[i].shape[0]), interpolation=cv2.INTER_NEAREST) # upsampled
    # Subtract the gaussian from upsampledd img(represents original img)
    sharpened = gaussian[i]- upsampled
    laplacian.append(sharpened) # Store the level

# Reconstruct the img
regenerated_img = laplacian[-1] # Start reconstructing from the smallest level
for i in range(len(laplacian) - 1, -1, -1):
    upsampled = cv2.resize(regenerated_img, (laplacian[i].shape[1], laplacian[i].shape[0]))
    # Upsampled to upper level and add it to the next level
    regenerated_img = cv2.add(upsampled, laplacian[i])

# Display original and regenerated img. 
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Img")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(regenerated_img, cv2.COLOR_BGR2RGB))
plt.title("Regenerated Img")

plt.tight_layout() 
plt.show()
