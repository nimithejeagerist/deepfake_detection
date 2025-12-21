from pathlib import Path

folder = Path("C:/Users/nimza/Desktop/deepfake_detection/FaceForensics++_C23/DeepFakes")
needle = "000"

matches = next(p for p in folder.iterdir() if p.name[:3] == needle)

print(matches)