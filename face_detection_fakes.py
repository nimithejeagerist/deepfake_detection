from typing import Tuple, Union
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_crop_coordinates(img, detection):
    width, height = img.shape[:2]
    
    bbox = detection.bounding_box
    x1 = bbox.origin_x 
    x2 = bbox.origin_x + bbox.width 
    y1 = bbox.origin_y 
    y2 = bbox.origin_y + bbox.height 
    
    if x2 > width or y2 > height:
        print("Warning: Crop area extends beyond image dimension")
    
    return x1, x2, y1, y2

IMAGE_FILE = "C:/Users/nimza/Documents/dd/fakes/1000.jpg"

img = cv2.imread(IMAGE_FILE)
# cv2.imshow("image", img)

base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

image = mp.Image.create_from_file(IMAGE_FILE)
detection_result = detector.detect(image)
detection = detection_result.detections[0]

x1, x2, y1, y2 = get_crop_coordinates(img, detection)
cropped_image = img[y1:y2, x1:x2]

cv2.imshow("cropped_image", cropped_image)
cv2.waitKey(0)