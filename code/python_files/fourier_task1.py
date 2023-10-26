import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Reference: https://stackoverflow.com/questions/52312053/how-to-combine-the-phase-of-one-image-and-magnitude-of-different-image-into-1-im
'''

# Read the images
img1 = cv2.imread('../input_images/cheetah.jpg') # Cheetah Image
img2 = cv2.imread('../input_images/cube_black.jpg') # Rubic cube Image

# Find fft of both images
fft_img1 = np.fft.fft2(img1, axes=(0, 1))
fft_img2 = np.fft.fft2(img2, axes=(0, 1))

# Find magnitude and phase of the ffts
mag1, phase1 = np.abs(fft_img1), np.angle(fft_img1)
mag2, phase2 = np.abs(fft_img2), np.angle(fft_img2)

# Apply logarithmic scaling to magnitude values
# I had to do this, because magnitude values were very big and I was not able to display them
log_mag1 = np.log(1 + mag1)
log_mag2 = np.log(1 + mag2)

# Normalize the scaled magnitude to get the values between [0, 1]
scaled_log_mag1 = (log_mag1 - np.min(log_mag1)) / (np.max(log_mag1) - np.min(log_mag1))
scaled_log_mag2 = (log_mag2 - np.min(log_mag2)) / (np.max(log_mag2) - np.min(log_mag2))

# Swapping phase of both imags
swapped_img1 = np.abs(mag1) * np.exp(1j * phase2)
swapped_img2 = np.abs(mag2) * np.exp(1j * phase1)

# Find ifft to regenerate the images
regenerated_img1 = np.fft.ifft2(swapped_img1, axes=(0, 1)).real.astype(np.uint8)
regenerated_img2 = np.fft.ifft2(swapped_img2, axes=(0, 1)).real.astype(np.uint8)

# Save the regenerated images
cv2.imwrite("../output_images/phase_swap1.png", regenerated_img1)
cv2.imwrite("../output_images/phase_swap2.png", regenerated_img2)

# Display all images
fig=plt.figure(figsize=(12, 12))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# Original Image 1
plt.subplot(3, 3, 1)
plt.title('Original Image 1')
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

# Original Image 2
plt.subplot(3, 3, 2)
plt.title('Original Image 2')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))


# scaled and logarithmic magnitude images
plt.subplot(3, 3, 4)
plt.title('Log Magnitude of Image 1')
plt.imshow(scaled_log_mag1, cmap='gray')

plt.subplot(3, 3, 5)
plt.title('Log Magnitude of Image 2')
plt.imshow(scaled_log_mag2, cmap='gray')


# Phase of Image 1
plt.subplot(3, 3, 7)
plt.title('Phase of Image 1')
plt.imshow(phase1, cmap='gray')

# Phase of Image 2
plt.subplot(3, 3, 8)
plt.title('Phase of Image 2')
plt.imshow(phase2, cmap='gray')

# Regenerated Image 1
plt.subplot(3, 3, 3)
plt.title('Regenerated Image 1')
plt.imshow(cv2.cvtColor(regenerated_img1, cv2.COLOR_BGR2RGB))

# Regenerated Image 2
plt.subplot(3, 3, 6)
plt.title('Regenerated Image 2')
plt.imshow(cv2.cvtColor(regenerated_img2, cv2.COLOR_BGR2RGB))

plt.show()
