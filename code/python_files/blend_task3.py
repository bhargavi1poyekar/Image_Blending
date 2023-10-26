import cv2
import numpy as np
import matplotlib.pyplot as plt

def blend(img1, img2, mask):

    # Decide depth to go to smallest level as possible
    smaller_dim = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    max_level = int(np.floor(np.log2(smaller_dim)))

    # Store the pyramid levels 
    gaussian_pyr = [img1]
    gaussian_pyr2 = [img2]
    mask_pyr = [mask]
    
    laplacian_pyr1 = []
    laplacian_pyr2 = []

    # Generate Gaussian and Laplacian pyramids for both images and the mask
    for _ in range(max_level):

        # Step 1=> Smoothen the image
        img1 = cv2.GaussianBlur(img1, (11, 11), 2)
        img2 = cv2.GaussianBlur(img2, (11, 11), 2)
        mask = cv2.GaussianBlur(mask, (11, 11), 2)

        # Downsample the images to half their sizes
        img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2))
        img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2))
        mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))

        # Store the pyramid levels
        gaussian_pyr.append(img1)
        gaussian_pyr2.append(img2)
        mask_pyr.append(mask)

    for i in range(len(gaussian_pyr) - 1): # At each pyramid level
        # Upsample the images
        upsampled1 = cv2.resize(gaussian_pyr[i + 1], (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
        upsampled2 = cv2.resize(gaussian_pyr2[i + 1], (gaussian_pyr2[i].shape[1], gaussian_pyr2[i].shape[0]))

        # Get the laplacian by subtracting smoothen image from reconstructed original version of that level.
        sharpened1 = cv2.subtract(gaussian_pyr[i], upsampled1)
        sharpened2 = cv2.subtract(gaussian_pyr2[i], upsampled2)

        # Store the levels. 
        laplacian_pyr1.append(sharpened1)
        laplacian_pyr2.append(sharpened2)

    # At each level perform alpha blending 
    alpha_blended_pyramid = []
    for i in range(max_level): 
        alpha = mask_pyr[i] / 255.0 # Normalize values between [0,1]
        alpha_blended = (1 - alpha) * laplacian_pyr1[i] + alpha * laplacian_pyr2[i]
        alpha_blended_pyramid.append(alpha_blended)

    # Combine the blended images to reconstruct the original blended image. 
    blended_image = alpha_blended_pyramid[-1]
    for i in range(max_level - 1, -1, -1):
        upsampled = cv2.resize(blended_image, (alpha_blended_pyramid[i].shape[1], alpha_blended_pyramid[i].shape[0]))
        blended_image = alpha_blended_pyramid[i] + upsampled

    return blended_image

# Read the images and mask
# img1 = cv2.imread("../input_images/orange.jpeg") 
# img2 = cv2.imread("../input_images/apple.jpeg")  
# mask = cv2.imread("../input_images/mask.png")   

# Read images for Task 4:
img1 = cv2.imread("../input_images/true_grit.jpg") 
img2 = cv2.imread("../input_images/chesapeake.jpg") 
mask = cv2.imread("../input_images/mask.png")

# Calling multiblend function
blended_image = blend(img1, img2, mask)

# Save the blended image as a 8-bit integer format
# cv2.imwrite("../output_images/orapple_blended_image.jpg", blended_image.astype(np.uint8))

cv2.imwrite("../output_images/grit_blended_image.jpg", blended_image.astype(np.uint8))

# Display the images
fig=plt.figure(figsize=(12, 4)) 
fig.subplots_adjust(hspace=0.5)

plt.subplot(131) 
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Image 1")

plt.subplot(132) 
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title("Image 2")

plt.subplot(133) 
plt.imshow(cv2.cvtColor(blended_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title("Blended image")

plt.tight_layout()
plt.show()
