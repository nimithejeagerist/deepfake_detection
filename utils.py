'''
Source of all helper functions
'''

from pathlib import Path

def stringify(number: int, path: str) -> str:
    folder = Path(path)
    target = None
    if number < 10:
        target = "00" + str(number)
    elif 10 <= number < 100:
        target = "0" + str(number)
    else:
        target = str(number)
    
    match = next(p for p in folder.iterdir() if p.name[:3] == target)
    return match

def stringify_new(number: int, path: str) -> str:
    folder = Path(path)
    target = None
    if number < 10:
        target = "000" + str(number)
    elif 10 <= number < 100:
        target = "00" + str(number)
    elif 100 <= number < 1000:
        target = "0" + str(number)
    else:
        target = str(number)
    
    if len(target) == 5:
        match = next(p for p in folder.iterdir() if p.name[:5] == target)
    else:
        match = next(p for p in folder.iterdir() if p.name[:4] == target)
    return match

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