# FACE_RECOGNITION

Simple face recognition training and inference utilities.

Files:
- `face_datasets.py` - dataset creation helper
- `training.py` - training script
- `face_recognition.py` - inference/recognition script
- `haarcascade_frontalface_default.xml` - face detector
- `names.csv` - label names
- `dataset/` - raw images organized by label
- `Trainer/` - generated trainer files (model outputs)

Quick start:
1. Prepare images under `dataset/<label>/` directories.
2. Run `training.py` to generate a model in `Trainer/`.
3. Use `face_recognition.py` for live recognition.

Dependencies:
- Python 3.8+
- opencv-python
- numpy

This repository was just initialized with a sensible `.gitignore`.
