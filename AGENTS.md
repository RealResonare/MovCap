# MovCap — Agent Instructions

## Project
Visual-Inertial Motion Capture System: 3 USB cameras + 8 BNO055 IMUs → BVH output.

## Tech Stack
- Python 3.10+ (primary), C++ (performance modules via pybind11)
- OpenCV, YOLO11-Pose (ultralytics), filterpy, bvhsdk, Open3D

## Code Style
- No comments unless asked
- Use type hints throughout
- Follow existing code conventions

## Key Commands
```bash
pip install -r requirements.txt
pytest tests/ -v
```
