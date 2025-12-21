# TODO: Import libaries
import os
import cv2
from utils import stringify

# Create PATH
path = "C:/Users/nimza/Documents/FaceForensics++_C23/DeepFakes"
saved_file_path = "C:/Users/nimza/Documents/dd/fakes/"

# TODO: Check if directories have been made and initialize
if not os.path.exists(saved_file_path):
    os.makedirs(saved_file_path)
  
# TODO: Open video file with OpenCV
file_count = 0
frame_count = 0
saved_frame_count = 0
interval = 15

while True:
    filename = stringify(file_count, path)
    file_count += 1
    cap = cv2.VideoCapture(filename)
    
    while True:
        if not os.path.exists(filename):
            break
        
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % interval == 0:
            cv2.imshow('frame', frame)
            break
    
    cap.release()
    cv2.destroyAllWindows()
    break
    
# TODO: Extract frames


# TODO: Save frames to fake folder at interval