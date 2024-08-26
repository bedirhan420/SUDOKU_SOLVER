print('Setting UP')

from utils.helpers import *
import cv2
import numpy as np
from sudoku_solver.solver import *
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# Path to the image
path_image = "Resources/1.jpg"
height_img = 450
width_img = 450

# Ensure the image file exists
if not os.path.exists(path_image):
    raise FileNotFoundError(f"Image file not found: {path_image}")

# Initialize model
model = initialize_prediction_model()

# Load and preprocess the image
img = cv2.imread(path_image)
img = cv2.resize(img, (width_img, height_img))
img_blank = np.zeros((height_img, width_img, 3), np.uint8)
img_threshold = preprocess(img)

# Find contours
img_contours = img.copy()
img_big_contour = img.copy()
contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

# Find and draw the biggest contour
biggest, max_area = biggest_contour(contours)
print(f"Biggest contour: {biggest}")

if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(img_big_contour, biggest, -1, (0, 0, 255), 25)
    
    # Perspective transformation
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp_colored = cv2.warpPerspective(img, matrix, (width_img, height_img))
    img_warp_colored = cv2.bitwise_not(img_warp_colored)

    img_detected_digits = img_blank.copy()
    img_warp_colored = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)

    # Split boxes and get predictions
    img_solved_digits = img_blank.copy()
    boxes = split_boxes(img_warp_colored)
    print(f"Number of boxes: {len(boxes)}")
    
    try:
        numbers = get_prediction(boxes, model)
    except Exception as e:
        print(f"Error during prediction: {e}")
        numbers = []
    
    print(f"Detected numbers: {numbers}")
    img_detected_digits = display_numbers(img_detected_digits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    pos_array = np.where(numbers > 0, 0, 1)
    
    # Solve Sudoku
    board = np.array_split(numbers, 9)
    try:
        solve(board)
    except Exception as e:
        print(f"Error during solving: {e}")
    
    # Prepare solved numbers
    flat_list = [item for sublist in board for item in sublist]
    solved_numbers = flat_list * pos_array
    img_solved_digits = display_numbers(img_solved_digits, solved_numbers)
    
    # Apply perspective transformation to solved digits
    pts2 = np.float32(biggest)
    pts1 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_in_warp_colored = img.copy()
    img_in_warp_colored = cv2.warpPerspective(img_solved_digits, matrix, (width_img, height_img))
    inv_perspective = cv2.addWeighted(img_in_warp_colored, 1, img, 0.5, 1)
    img_detected_digits = draw_grid(img_detected_digits)
    img_solved_digits = draw_grid(img_solved_digits)

    # Print debug information
    print(f"Numbers: {numbers}")
    print(f"Solved Numbers: {solved_numbers}")

    # Stack and display images
    image_array = ([img, img_threshold, img_contours, img_big_contour], 
                   [img_warp_colored, img_detected_digits, img_solved_digits, inv_perspective])
    stacked_image = stack_images(image_array, 0.8)
    print(f"Stacked Image shape: {stacked_image.shape}")
    cv2.imshow("Stacked Images", stacked_image)
else:
    print("No Sudoku Found")

cv2.waitKey(0)
cv2.destroyAllWindows()
