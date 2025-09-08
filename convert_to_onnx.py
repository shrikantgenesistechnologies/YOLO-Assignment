from ultralytics import YOLO
import argparse
import os

def convert_to_onnx(model_path, output_dir=None, imgsz=640, simplify=True, opset=12):
    """
    Convert a YOLO model to ONNX format
    
    Args:
        model_path: Path to the YOLO .pt model file
        output_dir: Directory to save the ONNX model (default: same directory as model_path)
        imgsz: Input image size for the model (default: 640)
        simplify: Whether to simplify the ONNX model (default: True)
        opset: ONNX opset version (default: 12)
    
    Returns:
        Path to the exported ONNX model
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Export the model to ONNX format
    print(f"Converting model to ONNX format...")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Export parameters - use different settings for better detection
    export_args = {
        'format': 'onnx',
        'imgsz': imgsz,
        'half': False,       # Use full precision
        'simplify': False,   # Don't simplify to preserve detection capabilities
        'opset': opset,      # ONNX opset version
        'dynamic': True,     # Dynamic axes
        'iou': 0.25,         # Lower IoU threshold for NMS to catch more overlapping objects
        'conf': 0.001,       # Very low confidence threshold for export to catch all potential detections
        'optimize': False,   # Don't optimize to preserve detection capabilities
        'nms': False,        # Disable NMS in the ONNX model to do it in post-processing
    }
    
    # Set the output path
    output_path = os.path.join(output_dir, f"{model_name}_full.onnx")
    
    # Export the model
    results = model.export(**export_args)
    
    # Get the path to the exported model
    onnx_path = results[0] if isinstance(results, list) else results
    
    print(f"Model successfully exported to: {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO model to ONNX format")
    parser.add_argument("--model", type=str, default="models/yolo11n.pt", help="Path to YOLO model file")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    
    args = parser.parse_args()
    
    convert_to_onnx(args.model, args.output, args.imgsz, args.simplify, args.opset)
