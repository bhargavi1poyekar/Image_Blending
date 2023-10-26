import cv2 # imports OpenCV
import numpy as np # this imports numpy
import matplotlib . pyplot as plt # imports matplotlib

################ Task 2: Loading and Displaying Image ####################

# Reads the image
elephant = cv2.imread("../input_images/elephant.jpeg")

# Read the saved image
elephant_opencv = cv2.imread("../output_images/elephant_opencv.png")

# Displays the image
cv2.imshow("Elephant Image", elephant)

# Converting image from BGR to RGB format
elephant_RGB= cv2.cvtColor(elephant_opencv, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(elephant_RGB)
plt.show()

# Save image
cv2.imwrite("../output_images/elephant_matplotlib.png", elephant_RGB)

# Waits for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

##################################################################

############### Task 3: Basic Image Processing ################## 

#### Grayscale #####

# Read image as grayscale
elephant_gray = cv2.imread("../input_images/elephant.jpeg", cv2.IMREAD_GRAYSCALE)

# Display the grayscale image using Matplotlib
# cmap => colormap
plt.imshow(elephant_gray, cmap='gray')
plt.show()

# Save the grayscale image
cv2.imwrite("../output_images/elephant_gray.png", elephant_gray)


##### Cropping #####

# Reads the image
elephant = cv2.imread("../input_images/elephant.jpeg")

x1, y1 = 0, 369
x2, y2 = 480, 997 

# Crop the image to get baby elephant
elephant_baby = elephant[y1:y2, x1:x2]

# Display the cropped baby elephant
plt.imshow(cv2.cvtColor(elephant_baby, cv2.COLOR_BGR2RGB))
plt.show()

# Save the baby elephant
cv2.imwrite("../output_images/elephant_baby.png", elephant_baby)


##### Resizing ######

# Read the image
elephant = cv2.imread("../input_images/elephant.jpeg")

# Convert the image to RGB
elephant_RGB = cv2.cvtColor(elephant, cv2.COLOR_BGR2RGB)

# Downsample the image by 10x in width and height 
elephant_downsampled = cv2.resize(elephant_RGB, None, fx=0.1, fy=0.1)

# Display the downsampled image
plt.imshow(elephant_downsampled)
plt.show()

# Save the downsampled image
cv2.imwrite("../output_images/elephant_10xdown.png", cv2.cvtColor(elephant_downsampled, cv2.COLOR_RGB2BGR))

# Interpolation Method: Nearest Neighbor
elphant_upsample_nearest = cv2.resize(elephant_downsampled, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)

# Interpolation Method: Bicubic
elphant_upsample_bicubic = cv2.resize(elephant_downsampled, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

# Save upsampled images
cv2.imwrite("../output_images/elephant_10xup_nearest.png", cv2.cvtColor(elphant_upsample_nearest, cv2.COLOR_RGB2BGR))
cv2.imwrite("../output_images/elephant_10xup_bicubic.png", cv2.cvtColor(elphant_upsample_bicubic, cv2.COLOR_RGB2BGR))

# Calculate absolute differences
nearest_difference = cv2.absdiff(elephant_RGB, elphant_upsample_nearest)
bicubic_difference = cv2.absdiff(elephant_RGB, elphant_upsample_bicubic)

# Save the difference images
cv2.imwrite("../output_images/elephant_10xup_nearest_diff.png", cv2.cvtColor(nearest_difference, cv2.COLOR_RGB2BGR))
cv2.imwrite("../output_images/elephant_10xup_bicubic_diff.png", cv2.cvtColor(bicubic_difference, cv2.COLOR_RGB2BGR))

# Calculate the sum of pixel differences
sum_nearest_diff = nearest_difference.sum()
sum_bicubic_diff = bicubic_difference.sum()

print("Nearest Neighbor error", sum_nearest_diff)
print("Bicubic error", sum_bicubic_diff)


##################################################################


