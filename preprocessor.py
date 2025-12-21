# TODO: Import libaries
import os
import cv2
from utils import stringify

# Create PATH
path = "C:/Users/nimza/Desktop/deepfake_detection/FaceForensics++_C23/DeepFakes/"
saved_file_path = "C:/Users/nimza/Desktop/deepfake_detection/fakes/"

# TODO: Check if directories have been made and initialize
if not os.exists(saved_file_path):
    os.makedirs(saved_file_path)
  
# TODO: Open video file with OpenCV
file_count = 0
frame_count = 0
saved_frame_count = 0
interval = 15

while True:
    stringified_file = 


# TODO: Extract frames


# TODO: Save frames to fake folder at interval