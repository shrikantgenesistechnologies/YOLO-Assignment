import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os
import time
import platform
from PIL import Image
from datetime import datetime
from tabulate import tabulate

def preprocess_image(image_path, input_size=(640, 640)):
    """
    Preprocess the image for ONNX model
    
    Args:
        image_path: Path to the image file
        input_size: Input size for the model (default: (640, 640))
    
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    original_height, original_width = img.shape[:2]
    
    # Calculate scale and padding
    scale = min(input_size[0] / original_width, input_size[1] / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    resized_img = cv2.resize(img, (new_width, new_height))
    
    # Calculate padding
    pad_width = input_size[0] - new_width
    pad_height = input_size[1] - new_height
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    
    # Add padding
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # Normalize and convert to NCHW format
    input_img = padded_img.astype(np.float32) / 255.0
    input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
    input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
    
    return input_img, (original_height, original_width), (top, left, scale)

def postprocess(outputs, img_shape, preprocess_info, conf_threshold=0.15, iou_threshold=0.45):
    """
    Postprocess the model outputs
    
    Args:
        outputs: Model outputs
        img_shape: Original image shape (height, width)
        preprocess_info: Preprocessing information (top padding, left padding, scale)
        conf_threshold: Confidence threshold (default: 0.15)
        iou_threshold: IOU threshold for NMS (default: 0.45)
    
    Returns:
        List of detections, each in format [x1, y1, x2, y2, confidence, class_id]
    """
    # Get predictions - YOLO v8 output format is [batch, num_classes+4, num_boxes]
    predictions = outputs[0]  # Shape: [1, 84, 8400]
    
    # Transpose to [batch, num_boxes, num_classes+4]
    predictions = predictions.transpose((0, 2, 1))  # Shape: [1, 8400, 84]
    predictions = predictions[0]  # Remove batch dimension, shape: [8400, 84]
    
    # Get box coordinates (first 4 values)
    boxes = predictions[:, :4]
    
    # Get confidence scores for each class (remaining values)
    scores = predictions[:, 4:]  # Shape: [8400, 80]
    
    # Process all detections first with standard confidence threshold
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    # Standard filtering
    mask = max_scores > conf_threshold
    boxes_filtered = boxes[mask]
    scores_filtered = scores[mask]
    class_ids_filtered = class_ids[mask]
    max_scores_filtered = max_scores[mask]
    
    # Now let's manually check for person detections with lower threshold
    person_threshold = 0.05  # Much lower threshold for persons
    
    # Find boxes with person class (class_id = 0) that have scores above the lower threshold
    person_scores = scores[:, 0]
    person_mask = person_scores > person_threshold
    
    # Filter out boxes that were already selected
    new_person_mask = person_mask.copy()
    for i in range(len(new_person_mask)):
        if mask[i]:  # If this box was already selected
            new_person_mask[i] = False
    
    # Get additional person detections
    if np.any(new_person_mask):
        additional_boxes = boxes[new_person_mask]
        additional_scores = person_scores[new_person_mask]
        # All these are person class (0)
        additional_class_ids = np.zeros(len(additional_boxes), dtype=np.int64)
        
        # Combine with existing detections
        boxes_final = np.vstack([boxes_filtered, additional_boxes]) if len(boxes_filtered) > 0 else additional_boxes
        max_scores_final = np.concatenate([max_scores_filtered, additional_scores]) if len(max_scores_filtered) > 0 else additional_scores
        class_ids_final = np.concatenate([class_ids_filtered, additional_class_ids]) if len(class_ids_filtered) > 0 else additional_class_ids
    else:
        boxes_final = boxes_filtered
        max_scores_final = max_scores_filtered
        class_ids_final = class_ids_filtered
    
    # Update variables for rest of function
    boxes = boxes_final
    max_scores = max_scores_final
    class_ids = class_ids_final
    
    if len(boxes) == 0:
        return []
    
    # Convert boxes to [x1, y1, x2, y2] format
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Rescale boxes to original image
    top, left, scale = preprocess_info
    original_height, original_width = img_shape
    
    x1 = (x1 - left) / scale
    y1 = (y1 - top) / scale
    x2 = (x2 - left) / scale
    y2 = (y2 - top) / scale
    
    # Clip boxes to image boundaries
    x1 = np.clip(x1, 0, original_width)
    y1 = np.clip(y1, 0, original_height)
    x2 = np.clip(x2, 0, original_width)
    y2 = np.clip(y2, 0, original_height)
    
    # Create detection array
    detections = np.column_stack((x1, y1, x2, y2, max_scores, class_ids))
    
    # Apply NMS (Non-Maximum Suppression)
    indices = cv2.dnn.NMSBoxes(
        detections[:, :4].tolist(),
        detections[:, 4].tolist(),
        conf_threshold,
        iou_threshold
    )
    
    if len(indices) > 0:
        return detections[indices].tolist()
    else:
        return []

def predict_with_onnx(model_path, image_path, conf_threshold=0.25, iou_threshold=0.45, verbose=True):
    """
    Use an ONNX YOLO model to predict objects in an image
    
    Args:
        model_path: Path to the ONNX model file
        image_path: Path to the image file
        conf_threshold: Confidence threshold (default: 0.25)
        iou_threshold: IOU threshold for NMS (default: 0.45)
        verbose: Whether to print detailed information (default: True)
    
    Returns:
        None (displays and saves the image with predictions)
    """
    start_time = time.time()
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Print header
    if verbose:
        print("\n" + "="*80)
        print(f"YOLO ONNX PREDICTION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # System info
        print(f"\n📊 SYSTEM INFORMATION:")
        print(f"  • OS: {platform.system()} {platform.release()}")
        print(f"  • Python: {platform.python_version()}")
        print(f"  • ONNX Runtime: {ort.__version__}")
        print(f"  • OpenCV: {cv2.__version__}")
        
        # Model and image info
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\n📁 FILES:")
        print(f"  • Model: {os.path.basename(model_path)} ({model_size:.2f} MB)")
        print(f"  • Image: {os.path.basename(image_path)}")
        
        # Parameters
        print(f"\n⚙️ PARAMETERS:")
        print(f"  • Confidence threshold: {conf_threshold}")
        print(f"  • IoU threshold: {iou_threshold}")
    
    # Load the ONNX model
    model_load_start = time.time()
    if verbose:
        print(f"\n🔄 Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    model_load_time = time.time() - model_load_start
    
    if verbose:
        print(f"  • Model loaded in {model_load_time:.2f} seconds")
        
        # Get model metadata
        input_details = session.get_inputs()[0]
        print(f"\n📋 MODEL DETAILS:")
        print(f"  • Input name: {input_details.name}")
        print(f"  • Input shape: {input_details.shape}")
        print(f"  • Input type: {input_details.type}")
    
    # Get model input name
    input_name = session.get_inputs()[0].name
    
    # Preprocess the image
    preprocess_start = time.time()
    if verbose:
        print(f"\n🖼️ Preprocessing image {image_path}...")
    input_img, img_shape, preprocess_info = preprocess_image(image_path)
    preprocess_time = time.time() - preprocess_start
    
    if verbose:
        print(f"  • Original dimensions: {img_shape[1]}x{img_shape[0]}")
        print(f"  • Input dimensions: {input_img.shape}")
        print(f"  • Preprocessing time: {preprocess_time*1000:.2f} ms")
    
    # Run inference
    inference_start = time.time()
    if verbose:
        print(f"\n🧠 Running inference...")
    outputs = session.run(None, {input_name: input_img})
    inference_time = time.time() - inference_start
    
    if verbose:
        print(f"  • Inference time: {inference_time*1000:.2f} ms")
        print(f"  • FPS: {1/inference_time:.2f}")
    
    # Postprocess the outputs
    postprocess_start = time.time()
    if verbose:
        print(f"\n🔍 Postprocessing results...")
    detections = postprocess(outputs, img_shape, preprocess_info, conf_threshold, iou_threshold)
    postprocess_time = time.time() - postprocess_start
    
    if verbose:
        print(f"  • Postprocessing time: {postprocess_time*1000:.2f} ms")
    
    # Load class names (COCO classes)
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    
    # Load the original image for drawing
    img = cv2.imread(image_path)
    
    # Draw detections
    detection_count = {}
    detection_details = []
    
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)
        
        # Get class name
        class_name = class_names[class_id]
        
        # Count detections by class
        if class_name in detection_count:
            detection_count[class_name] += 1
        else:
            detection_count[class_name] = 1
        
        # Store detection details for table
        width = x2 - x1
        height = y2 - y1
        area = width * height
        detection_details.append([
            class_name, 
            f"{confidence:.4f}", 
            f"{x1},{y1}", 
            f"{x2},{y2}", 
            f"{width}x{height}", 
            f"{area}"
        ])
        
        # Draw bounding box (color based on confidence)
        # Higher confidence = more green, lower confidence = more red
        green = min(255, int(confidence * 255))
        red = min(255, int((1 - confidence) * 255))
        color = (0, green, red)  # BGR format
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add label with confidence score
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save the result
    output_path = f"output_onnx_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, img)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    if verbose:
        # Print detection results
        print(f"\n🎯 DETECTION RESULTS:")
        
        if len(detections) > 0:
            # Print detailed detection table
            print("\nDetailed Detections:")
            headers = ["Class", "Confidence", "Top-Left", "Bottom-Right", "Dimensions", "Area"]
            print(tabulate(detection_details, headers=headers, tablefmt="grid"))
            
            # Print summary by class
            print("\nDetection Summary:")
            summary_table = [[class_name, count] for class_name, count in detection_count.items()]
            print(tabulate(summary_table, headers=["Class", "Count"], tablefmt="simple"))
            
            # Print total objects
            total_objects = sum(detection_count.values())
            print(f"\nTotal objects detected: {total_objects}")
        else:
            print("  • No objects detected")
        
        # Print timing summary
        print(f"\n⏱️ TIMING SUMMARY:")
        print(f"  • Preprocessing: {preprocess_time*1000:.2f} ms")
        print(f"  • Inference: {inference_time*1000:.2f} ms")
        print(f"  • Postprocessing: {postprocess_time*1000:.2f} ms")
        print(f"  • Total time: {total_time*1000:.2f} ms")
        print(f"  • FPS: {1/total_time:.2f}")
        
        print(f"\n✅ Prediction complete. Result saved to {output_path}")
        print("="*80)
    else:
        print(f"Prediction complete. Result saved to {output_path}")
        
        # Display simple summary of detections
        print("\nDetection Summary:")
        for class_name, count in detection_count.items():
            print(f"- {count} {class_name}{'s' if count > 1 else ''}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO ONNX object detection on images")
    parser.add_argument("--model", type=str, default="models/yolo11n.onnx", help="Path to YOLO ONNX model file")
    parser.add_argument("--image", type=str, default="images/image-2.png", help="Path to image file")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    
    args = parser.parse_args()
    
    predict_with_onnx(args.model, args.image, args.conf, args.iou, verbose=not args.quiet)
