# TODO: Import libaries
import os
import cv2
from utils import stringify

# Create PATH
path = "C:/Users/nimza/Documents/FaceForensics++_C23/original"
saved_file_path = "C:/Users/nimza/Documents/dd/real/"

# TODO: Check if directories have been made and initialize
if not os.path.exists(saved_file_path):
    os.makedirs(saved_file_path)
  
# TODO: Open video file with OpenCV
file_count = 0
frame_count = 0
saved_frame_count = 0
interval = 24

while True:
    filename = stringify(file_count, path)
    file_count += 1
    
    if not os.path.exists(filename):
        break
    
    cap = cv2.VideoCapture(filename)
    
    while True:
        if not os.path.exists(filename):
            break
        
        # TODO: Extract frames
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # TODO: Save frames to fake folder at interval
        if frame_count % interval == 0:
            filename = os.path.join(saved_file_path, f"{saved_frame_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved frame {saved_frame_count}.jpg")
            saved_frame_count += 1
            
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()