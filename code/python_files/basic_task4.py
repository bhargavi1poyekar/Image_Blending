import cv2
import numpy as np
import matplotlib.pyplot as plt

################ Task 4: Edge Detection and Image Blurring ####################

# Edge Detection
'''
Reference: https://stackoverflow.com/questions/51167768/sobel-edge-detection-using-opencv
'''

def edge_detection(image):
   
    # Read the image
    elephant = image

    # Sobel filter for gradients in x-direction
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    
    # Sobel filter for gradients in y_direction
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    
    # Sobel filter for both horizontal and vertical edges
    edge_vertical = cv2.filter2D(elephant, -1, sobel_x) 
    # Parameters=> (image, depth, kernel) => depth: -1 output image depth same as input image.
    edge_horizontal = cv2.filter2D(elephant, -1, sobel_y)
    
    # Combine the results to detect edges in all directions
    edge_output = cv2.addWeighted(edge_vertical, 0.5, edge_horizontal, 0.5, 0)
    
    return edge_output

# Image Blurring
'''
Reference: Slides of our lecture for Gaussian Filter. 
''' 
def image_blurring(image):
    
    # Gaussian filter
    gauss_filter = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]) / 16  
    
    # Apply filter
    blurred_output = cv2.filter2D(image, -1, gauss_filter)
    
    return blurred_output

# Display the 3 images 
def display(original_image, edge_output, blurred_output):
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(edge_output, cv2.COLOR_BGR2RGB))
    plt.title("Edge Detection")
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(blurred_output, cv2.COLOR_BGR2RGB))
    plt.title("Blurring (Smoothing)")
    
    plt.tight_layout()
    plt.show()

elephant = cv2.imread('../input_images/elephant.jpeg')

# Edge detection
edge_output = edge_detection(elephant)

# Image blurring
blurred_output = image_blurring(elephant)

cv2.imwrite("../output_images/elephant_edge_output.png", edge_output)
cv2.imwrite("../output_images/elephant_blurred_output.png", blurred_output)

# Display Images
display(elephant, edge_output, blurred_output)


