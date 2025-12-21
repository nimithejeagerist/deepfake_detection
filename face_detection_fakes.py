import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import stringify_new, get_crop_coordinates

path = "C:/Users/nimza/Documents/dd/fakes"
saved_file_path = "C:/Users/nimza/Documents/dd/fakes_cropped/"

if not os.path.exists(saved_file_path):
    os.makedirs(saved_file_path)

base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

saved_frame_count = 0

while True:
    IMAGE_FILE = stringify_new(saved_frame_count)
    
    img = cv2.imread(IMAGE_FILE)

    image = mp.Image.create_from_file(IMAGE_FILE)
    detection_result = detector.detect(image)
    detection = detection_result.detections[0]

    x1, x2, y1, y2 = get_crop_coordinates(img, detection)
    cropped_image = img[y1:y2, x1:x2]
    
    

    