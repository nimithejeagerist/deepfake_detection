from fastapi import FastAPI, UploadFile, File
import os
import shutil
import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms
from .model import ResNet34Original
from fastapi.middleware.cors import CORSMiddleware

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(__file__)

TEMP_DIR = os.path.join(BASE_DIR, "temp")
FACE_DIR = os.path.join(TEMP_DIR, "faces")
VIDEO_PATH = os.path.join(TEMP_DIR, "video.mp4")

MODEL_PATH = os.path.join(BASE_DIR, "resnet34_best.pt")
DETECTOR_PATH = os.path.join(BASE_DIR, "detector.tflite")

THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # your Next.js app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(FACE_DIR, exist_ok=True)

# ---- MediaPipe Face Detector (explicit TFLite) ----
base_options = python.BaseOptions(model_asset_path=DETECTOR_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
face_detector = vision.FaceDetector.create_from_options(options)

# ---- Load classifier once ----
model = ResNet34Original(in_channels=3, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---- Image preprocessing ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

def get_crop_coordinates(img, detection):
    """Replica of your utils.get_crop_coordinates logic"""
    h, w, _ = img.shape
    bbox = detection.bounding_box

    x1 = max(0, bbox.origin_x)
    y1 = max(0, bbox.origin_y)
    x2 = min(w, bbox.origin_x + bbox.width)
    y2 = min(h, bbox.origin_y + bbox.height)

    return x1, x2, y1, y2


@app.post("/detect")
async def detect(video: UploadFile = File(...)):
    # ---- reset temp folders ----
    if os.path.exists(FACE_DIR):
        shutil.rmtree(FACE_DIR)
    os.makedirs(FACE_DIR, exist_ok=True)

    # ---- save video ----
    with open(VIDEO_PATH, "wb") as f:
        f.write(await video.read())

    # ---- extract faces from frames ----
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = 10
    stride = max(total_frames // max_frames, 1)
    print(stride)
    frame_idx = 0
    sampled = 0

    try:
        while sampled < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
              
            # only process every `stride`-th frame
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
              
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = face_detector.detect(mp_image)

            if detection_result.detections:
                detection = detection_result.detections[0]

                x1, x2, y1, y2 = get_crop_coordinates(frame, detection)
                face = frame[y1:y2, x1:x2]

                if face.size != 0:
                    cv2.imwrite(
                        os.path.join(FACE_DIR, f"face_{sampled:04d}.jpg"),
                        face
                    )
                    sampled += 1

            frame_idx += 1
    finally:
        cap.release()


    # ---- load faces into tensors + infer ----
    face_probs = []

    for fname in os.listdir(FACE_DIR):
        img_path = os.path.join(FACE_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            fake_prob = probs[0, 1].item()
            face_probs.append(fake_prob)

    # ---- aggregate ----
    if len(face_probs) == 0:
        avg_fake_prob = 0.0
    else:
        avg_fake_prob = float(np.mean(face_probs))

    label = "fake" if avg_fake_prob >= THRESHOLD else "real"
    confidence = avg_fake_prob if label == "fake" else 1.0 - avg_fake_prob

    return {
        "label": label,
        "confidence": confidence,
    }