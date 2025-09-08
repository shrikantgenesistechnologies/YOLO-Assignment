# YOLO Object Detection Assignment

This project demonstrates object detection using YOLO (You Only Look Once) models in both PyTorch and ONNX formats. The implementation includes scripts for prediction, model conversion, and a hybrid approach that combines the strengths of both formats.

## Project Structure

```
YOLO-assignment/
├── images/                  # Input images for testing
│   └── image-2.png
├── models/                  # Model files
│   ├── yolo11n.pt           # PyTorch model
│   └── yolo11n.onnx         # ONNX model (converted from PyTorch)
├── convert_to_onnx.py       # Script to convert PyTorch model to ONNX
├── predict.py               # Script for prediction using PyTorch model
├── predict_onnx.py          # Script for prediction using ONNX model
├── hybrid_predict.py        # Script for prediction using both models
├── output_image-2.png       # PyTorch model prediction output
├── output_onnx_image-2.png  # ONNX model prediction output
└── output_hybrid_image-2.png # Hybrid approach prediction output
```

## Setup and Installation

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Setting Up Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install torch torchvision ultralytics onnx onnxruntime tabulate
```

## Usage

### 1. PyTorch Model Prediction

The `predict.py` script uses the PyTorch YOLO model for object detection:

```bash
python predict.py --model models/yolo11n.pt --image images/image-2.png --conf 0.25
```

Options:
- `--model`: Path to the PyTorch model file (default: models/yolo11n.pt)
- `--image`: Path to the image file (default: images/image-2.png)
- `--conf`: Confidence threshold (default: 0.25)

### 2. Converting PyTorch Model to ONNX

The `convert_to_onnx.py` script converts a PyTorch YOLO model to ONNX format:

```bash
python convert_to_onnx.py --model models/yolo11n.pt --simplify
```

Options:
- `--model`: Path to the PyTorch model file (default: models/yolo11n.pt)
- `--output`: Output directory (default: same directory as model)
- `--imgsz`: Input image size (default: 640)
- `--simplify`: Simplify the ONNX model (flag)
- `--opset`: ONNX opset version (default: 12)

### 3. ONNX Model Prediction

The `predict_onnx.py` script uses the ONNX YOLO model for object detection with detailed console output:

```bash
python predict_onnx.py --model models/yolo11n.onnx --image images/image-2.png --conf 0.15
```

Options:
- `--model`: Path to the ONNX model file (default: models/yolo11n.onnx)
- `--image`: Path to the image file (default: images/image-2.png)
- `--conf`: Confidence threshold (default: 0.15)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--quiet`: Disable verbose output (flag)

### 4. Hybrid Prediction (PyTorch + ONNX)

The `hybrid_predict.py` script combines both PyTorch and ONNX models for optimal detection:

```bash
python hybrid_predict.py --pt-model models/yolo11n.pt --onnx-model models/yolo11n.onnx --image images/image-2.png
```

Options:
- `--pt-model`: Path to the PyTorch model file (default: models/yolo11n.pt)
- `--onnx-model`: Path to the ONNX model file (default: models/yolo11n.onnx)
- `--image`: Path to the image file (default: images/image-2.png)
- `--conf`: Confidence threshold (default: 0.15)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--quiet`: Disable verbose output (flag)

## Features

### PyTorch Model (`predict.py`)
- Basic object detection
- Simple console output
- Output image with bounding boxes

### ONNX Model (`predict_onnx.py`)
- Enhanced console output with detailed information
- System information display
- Performance metrics (inference time, FPS)
- Tabular detection results
- Color-coded bounding boxes based on confidence

### Hybrid Approach (`hybrid_predict.py`)
- Uses PyTorch model for person detection (better at detecting small people)
- Uses ONNX model for other object classes
- Color-coded bounding boxes (red for PyTorch detections, green/blue for ONNX)
- Comprehensive console output
- Best detection results by combining strengths of both models

## Key Findings

1. The PyTorch model generally has better detection capabilities for smaller objects, particularly people.
2. The ONNX model offers better deployment options and cross-platform compatibility.
3. The hybrid approach provides the best of both worlds by using:
   - PyTorch for person detection
   - ONNX for other object classes

## Console Output Example

The enhanced console output includes:
- System information
- Model details
- Input/output file information
- Performance metrics
- Detailed detection results in tabular format
- Detection summary

## Troubleshooting

If you encounter issues with person detection in the ONNX model:
1. Try lowering the confidence threshold
2. Use the hybrid approach which leverages PyTorch's better person detection
3. Experiment with different ONNX conversion parameters

## License

This project is provided for educational purposes only.
