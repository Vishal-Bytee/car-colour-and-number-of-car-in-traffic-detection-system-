# Car Colour Detection System
A machine learning model that detects and classifies cars by colour in traffic 
images, counts people at traffic signals, and uses colour-coded bounding boxes 
based on the detected car colour.

## Features
- Detects multiple cars in a single frame
- YOLOv8 pre-trained model for accurate vehicle detection
- RED boxes for blue coloured cars
- BLUE boxes for all other car colours
- GREEN boxes for people/pedestrians
- Counts total cars and people at traffic signal
- GUI with side-by-side input and result preview
- Save annotated result as image

## What It Can Detect

### Vehicles
-- Type 
Car 
Truck 
Bus 
Motorcycle 

### People
 Type 
Pedestrian / Person 

## Colour Classification

| Detected Colour | Box Colour |
| Blue car | RED box |
| Red car | BLUE box |
| White car | BLUE box |
| Black car | BLUE box |
| Silver car | BLUE box |
| Yellow car | BLUE box |
| Green car | BLUE box |

## How It Works


Image Input
      |
YOLOv8 ML Model (yolov8n.pt)
      |
Detect Vehicles + People + Confidence Score
      |
Classify Car Colour using HSV Analysis
      |
RED Box → Blue Car
BLUE Box → Other Colour Car
GREEN Box → Person
      |
Show Stats Panel (car count, colour breakdown, people count)
      |
Save Output Image


## Technologies Used

| Tool | Purpose |

| Python 3.8+ | Core programming language |
| YOLOv8 (ultralytics) | Vehicle and person detection ML model |
| OpenCV | Image processing and colour classification |
| Tkinter | GUI framework |
| Pillow | Image format conversion for display |
| NumPy | Numerical operations |

## System Requirements

- Python 3.8 or higher
- RAM: 4GB minimum (8GB recommended)
- Storage: 500MB free space (for model weights)
- OS: Windows 10/11, Linux, macOS
- Internet connection on first run (to download model weights)

## Installation

### 1. Clone Repository
bash
git clone https://github.com/Vishal-Bytee/car-colour-and-number-of-car-in-traffic-detection-system.git
cd car-colour-detection




### 2. Install Dependencies
bash
pip install -r requirements.txt


## How to Run
bash
python app.py


### For Image Detection
1. Click **Load Image**
2. Select image file (.jpg, .png, .bmp)
3. Click **Analyse**
4. View results with colour-coded bounding boxes
5. Check stats panel for car count and colour breakdown

### Saving Results
1. After analysing, click **Save Result**
2. Choose location and filename
3. Annotated image saved as .jpg or .png

## Project Structure

car-colour-detection/

─ app.py                     # Main GUI application
─ test_detection.py          # Headless test script
─ requirements.txt           # Dependencies
─ README.md                  # Documentation

 utils/
    ─ model_loader.py        # Loads YOLOv8 model
    ─ detector.py            # Runs detection and draws boxes
    ─ color_classifier.py   # HSV-based colour classification


## Limitations
- Works best with clear, well-lit traffic images
- Very small or distant cars may be missed
- Similar colours (silver/white) may occasionally be confused
- Only classifies the 7 trained colour categories
- First run requires internet to download model weights (~6MB)

## Future Improvements
-  Real-time webcam / live traffic feed support
-  Video file processing
-  GPU acceleration for faster detection
-  More colour categories
-  Export detection report as PDF
- [ ] Night-time / low-light image support
