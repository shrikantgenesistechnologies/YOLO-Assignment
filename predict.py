from ultralytics import YOLO
import cv2
import os
import argparse

def predict_image(model_path, image_path, conf_threshold=0.25):
    """
    Use a YOLO model to predict objects in an image
    
    Args:
        model_path: Path to the .pt model file
        image_path: Path to the image file
        conf_threshold: Confidence threshold for predictions (default: 0.25)
    
    Returns:
        None (displays and saves the image with predictions)
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run inference on the image
    print(f"Running prediction on {image_path}...")
    results = model(image_path, conf=conf_threshold)
    
    # Get the original image
    img = cv2.imread(image_path)
    
    # Process results
    for result in results:
        boxes = result.boxes
        
        # Plot bounding boxes and labels on the image
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Get class name
            class_name = result.names[cls]
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the result
    output_path = f"output_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, img)
    print(f"Prediction complete. Result saved to {output_path}")
    
    # Display summary of detections
    for result in results:
        print("\nDetection Summary:")
        for c in result.boxes.cls.unique():
            n = (result.boxes.cls == c).sum()
            class_name = result.names[int(c)]
            print(f"- {n} {class_name}{'s' if n > 1 else ''}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO object detection on images")
    parser.add_argument("--model", type=str, default="models/yolo11n.pt", help="Path to YOLO model file")
    parser.add_argument("--image", type=str, default="images/image-2.png", help="Path to image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    predict_image(args.model, args.image, args.conf)
