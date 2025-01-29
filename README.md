# Soccer Match Analysis with YOLO

## Overview
This project uses You Only Look Once (YOLO)-based Object Detection and Object Tracking to detect, track, and analyze professional soccer match footage. By detecting players, referees, and the ball, it provides detailed insights into player movement, ball control, and game statistics. The project is built with OpenCV, Ultralytics YOLOv5, and Supervision libraries and was based on a tutorial from YouTube tutorial from CodeInAJiffy on YouTube.

---

## Features
- **Player, Referee, and Ball Detection**
  - Trained using a RoboFlow dataset with 612 images.
  - Detection via YOLOv5 model using transfer learning.
- **Object Tracking**
  - Tracks players and referees, assigning unique IDs.
- **Camera Movement Compensation**
  - Corrects for camera motion using optical flow.
- **Perspective Transformation**
  - Transforms the view into a top-down rectangular field.
- **Speed and Distance Estimation**
  - Calculates speed and total distance covered by players.
- **Team and Player Ball Control**
  - Tracks ball possession and calculates team control percentages.
- **Jersey-Based Team Assignment**
  - Assigns players to teams based on jersey colors using K-Means clustering.
- **Visual Annotations**
  - Displays positions, speeds, ball control stats, and camera motion on video.

---

## Workflow

### 1. Video Frame Processing
- Frames are read using OpenCV's `cv2.VideoCapture`.

### 2. Object Detection
- YOLOv5 detects players, referees, and the ball in batches of 20 frames for efficiency.

### 3. Object Tracking
- Tracks objects across frames, assigning unique IDs to players and referees.

### 4. Camera Movement Estimation
- Optical flow calculates camera motion based on non-field features like stadium corners.

### 5. Perspective Transformation
- Transforms the video to a top-down rectangular view of the field.

### 6. Ball Position Interpolation
- Fills missing ball detections using **pandas** interpolation.

### 7. Speed and Distance Analysis
- Converts tracked positions to real-world distances and calculates speed and distance over time.

### 8. Team Assignment
- Classifies players into teams based on jersey colors using K-Means clustering.

### 9. Ball Control Statistics
- Tracks ball possession per player and team, calculating control percentages.

### 10. Video Annotations
- Visualizes:
  - Player positions and IDs.
  - Ball position and control.
  - Player speeds and distances.
  - Camera movement (x, y pixel shifts).
  - Team ball control statistics.

---

## Tools and Libraries
- **YOLOv5 (Ultralytics):** For object detection and tracking.
- **OpenCV:** For video processing, optical flow, and perspective transformations.
- **Supervision:** For tracking and assigning IDs to detected objects.
- **pandas:** For ball position interpolation and data analysis.
- **K-Means Clustering:** For team assignment based on jersey colors.

---

## Future Improvements
1. **Improved Ball Detection**
   - Use a better dataset and larger input frames for training.
2. **Goalkeeper Tracking**
   - Retain goalkeeper classification for advanced analysis.
3. **Expanded Dataset**
   - Increase dataset size and diversity for better generalization.
