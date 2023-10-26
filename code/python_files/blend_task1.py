import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images and mask - Orange and Apple
# img1 = cv2.imread("../input_images/orange.jpeg") 
# img2 = cv2.imread("../input_images/apple.jpeg") 
# mask = cv2.imread("../input_images/mask.png") 

# Read images for Task 4:
img1 = cv2.imread("../input_images/true_grit.jpg") 
img2 = cv2.imread("../input_images/chesapeake.jpg") 
mask = cv2.imread("../input_images/mask.png") 

# Convert the images to 8-bit integers (Scales to values between 0-255)
img1 = cv2.convertScaleAbs(img1)
img2 = cv2.convertScaleAbs(img2)

# Direct blending
Direct_blend = (1 - mask / 255) * img1 + (mask / 255) * img2
# /255=> Normalize the values => [0,1]

# Convert the direct blend result to 8-bit integers
Direct_blend = cv2.convertScaleAbs(Direct_blend)

# Blurring the mask edge with kernel size 15,15 => More the kernel size, better the smoothening effect
mask_filtered = cv2.GaussianBlur(mask, (15, 15), 8)

# Alpha blending
Alpha_blend = (1 - mask_filtered / 255) * img1 + (mask_filtered / 255) * img2

Alpha_blend = cv2.convertScaleAbs(Alpha_blend)

# cv2.imwrite("../output_images/orapple_direct_blend.png", Direct_blend)
# cv2.imwrite("../output_images/orapple_alpha_blend.png", Alpha_blend)

cv2.imwrite("../output_images/grit_direct_blend.png", Direct_blend)
cv2.imwrite("../output_images/grit_alpha_blend.png", Alpha_blend)

# Display the images
fig=plt.figure(figsize=(20, 4)) 
fig.subplots_adjust(hspace=0.5)

plt.subplot(151) 
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Image 1")

plt.subplot(152) 
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title("Image 2")

plt.subplot(153) 
plt.imshow(mask, cmap='gray')
plt.title("Mask (M)")

plt.subplot(154) 
plt.imshow(cv2.cvtColor(Direct_blend, cv2.COLOR_BGR2RGB))
plt.title("Direct Blend")

plt.subplot(155) 
plt.imshow(cv2.cvtColor(Alpha_blend, cv2.COLOR_BGR2RGB))
plt.title("Alpha Blend")



plt.tight_layout()
plt.show()
