# Deepfake Detector
A lightweight end-to-end system for detecting manipulated videos using face-level visual cues and a trained convolutional neural network.

This project was built as an exercise in model deployment, inference pipelines, and system integration, rather than as a production-ready detection service.
## Demo
*(add video or GIF here)*

## Motivation
Deepfake detection is often discussed in abstract terms, but deploying even a simple detector reveals practical questions that are easy to overlook:

- How many frames should a model actually look at?
- How should per-frame predictions be aggregated into a single decision?
- How do preprocessing choices affect inference reliability?
- Where do latency and bottlenecks appear in a real pipeline?

The goal of this project was not to build the “best” detector, but to translate a trained model into a working system that makes clear, interpretable decisions on real video inputs.

## System Overview
The system is composed of two main components:
- **Frontend (Next.js)**  
  Provides a minimal interface for uploading a video, previewing it, and running detection.  
  The UI is intentionally simple, focusing on clarity rather than explanation or persuasion.

- **Backend (FastAPI + PyTorch + MediaPipe)**  
  Handles video ingestion, frame sampling, face detection, model inference, and result aggregation.

All model inference runs locally. Videos are processed transiently and are not persisted after inference.

## Detection Pipeline
At a high level, the backend performs the following steps:
1. **Video ingestion**  
   The uploaded video is temporarily stored and read frame-by-frame.
2. **Frame sampling**  
   A small, fixed number of frames (at most 10) are sampled evenly across the video to limit latency and avoid over-processing long clips.
3. **Face detection**  
   Faces are detected using a MediaPipe TFLite model and cropped from each sampled frame.
4. **Model inference**  
   Each cropped face is resized to a fixed resolution and passed through a trained ResNet-based classifier.
5. **Aggregation**  
   Per-face predictions are averaged to produce a single video-level confidence score.
6. **Decision**  
   The final label (“real” or “fake”) is determined via a simple threshold on the aggregated confidence.

This design keeps the pipeline transparent and easy to reason about.

## Design Considerations
Several deliberate choices shaped the system:
- **Frame-level reasoning**  
  The model operates on individual face crops rather than full videos, keeping inference simple and interpretable.
- **Controlled sampling**  
  Limiting the number of frames avoids unnecessary computation and makes latency predictable.
- **Aggregation over explanation**  
  The system reports a final label and confidence, without attempting to justify individual predictions.
- **Local inference**  
  Running everything locally exposes real constraints around preprocessing consistency, batching, and performance.
- **Ephemeral data handling**  
  Videos and extracted frames are deleted after processing, keeping the system stateless.

## What This Demonstrates
This project focuses less on model novelty and more on practical deployment concerns, including:
- bridging trained models and real inputs,
- handling video data in a live system,
- managing preprocessing consistency between training and inference,
- reasoning about aggregation and thresholds,
- building simple, reliable interfaces for model-driven systems.

It is intentionally minimal, but aims to be conceptually honest about what the detector does and does not claim to solve.
