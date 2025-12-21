import shutil
import os
import cv2

# Create PATH
path = "C:/Users/nimza/Desktop/deepfake_detection/demo.mp4"
saved_file_path = "C:/Users/nimza/Desktop/deepfake_detection/saved_pictures/"

shutil.rmtree(saved_file_path)
os.makedirs(saved_file_path)
    
# Read a video from a file
cap = cv2.VideoCapture(path)

if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video file opened successfully!")
    
frame_count = 0
saved_frame_count = 0
frame_interval = 20

while True:
    # Read the first frame to confirm reading
    ret, frame = cap.read()
    
    if not ret:
        break
    
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(saved_file_path, f"frame_{saved_frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1
        print(f"Saved: Frame {saved_frame_count}")
        
    frame_count += 1
    

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
print(f"\nExtraction complete. Total frames saved: {saved_frame_count}")