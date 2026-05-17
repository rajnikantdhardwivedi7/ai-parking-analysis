# Parking Management System — Aerial/Top-Down Edition

## Overview
This project is an automated Parking Management System designed for aerial or top-down video footage. It uses computer vision techniques (background subtraction, contour detection, and YOLO object detection) to detect cars in parking lots from drone or overhead camera footage. The system analyzes parking slot occupancy and produces a processed video with visual overlays indicating parked and available spots.

## Features
- Detects moving and parked cars using OpenCV and YOLOv8
- Automatically identifies parking slots based on video layout
- Visualizes occupancy status and statistics on the output video
- Outputs a compressed video with overlays

## How to Fork and Run
1. **Forking the Repository**
   - Click the "Fork" button on the top right of the GitHub repository page to create your own copy.
   - Clone your forked repository to your local machine:
     ```bash
     git clone https://github.com/your-username/your-forked-repo.git
     cd your-forked-repo
     ```

2. **Running the Project**
   - Place your aerial/top-down parking lot video (e.g., `input.mp4`) in the project directory.
   - Run the script using Python 3.11 or later:
     ```bash
     python main.py input.mp4
     ```
   - The output video with overlays will be saved as `output.mp4`.

## Relevant Wikipedia Links
- [Parking lot](https://en.wikipedia.org/wiki/Parking_lot)
- [Computer vision](https://en.wikipedia.org/wiki/Computer_vision)
- [YOLO (object detection)](https://en.wikipedia.org/wiki/You_Only_Look_Once)
- [Background subtraction](https://en.wikipedia.org/wiki/Background_subtraction)
- [Contour detection](https://en.wikipedia.org/wiki/Contour_(image_processing))

## Developer

## rajnikantdhardwivedi
