'''
Source of all helper functions
'''

import Path

def stringify(number: int, path: str) -> str:
    folder = Path(path)
    target = None
    if number < 10:
        target = "00" + str(number)
    elif 10 <= number < 100:
        target = "0" + str(number)
    else:
        target = str(number)
    
    match = next()