from typing import Tuple, Union
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.detections import Detection

MARGIN = 10 #pixels
ROW_SIZE = 10 #pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0) #red

def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
    """
    Converts normalized value pair to pixel coordinates.
    """
    
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))
    
    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        return None
    
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    
def visualize(image, detection_result) -> np.ndarray:
    """
    Draws bounding boxes and keypoints on the input image and return it
    Args:
     image: The input RGB image
     detection_result: The list of all "Detection" entities to visualize.
    Returns:
     Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape
    
    for detection in detection_result.detections:
        # Draw bouding box
        bbox = detection.bounding_box
        starting_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, starting_point, end_point, TEXT_COLOR, 3)
        print(get_crop_coordinates(detection))

        # Draw keypoints
        if detection.keypoints == None:
            return annotated_image
        
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
        
        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + '(' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
        
    return annotated_image

def get_crop_coordinates(detection: Detection) -> list:
    bbox = detection.bounding_box
    x1 = bbox.origin_x - 10
    x2 = bbox.origin_x + bbox.width + 10
    y1 = bbox.origin_y - 10
    y2 = bbox.origin_y + bbox.height + 10
    return [x1, x2, y1, y2]

IMAGE_FILE = "C:/Users/nimza/Documents/dd/fakes/1000.jpg"

img = cv2.imread(IMAGE_FILE)
# cv2.imshow("image", img)
print(img.shape)

base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

image = mp.Image.create_from_file(IMAGE_FILE)

detection_result = detector.detect(image)
print(type(detection_result.detections[0]))

image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2.imshow("rgb_image", rgb_annotated_image)
cv2.waitKey(0)