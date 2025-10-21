##############################################
# @Author: Evan Rapoza
# @Contributor(s): Peter Nolan
# @Document: 'object_detector.py'
#
# Description:
# This module performs object detection on video frames using
# a pretrained YOLOv8 model. Using object orientated design,
# it returns bounding boxes, labels, and confidence
# scores for each detected object.
#
##############################################

import numpy as np
import cv2
from ultralytics import YOLO
from typing import List, Dict

class ObjectDetector:

    # Initialize OOD
    def __init__(self, modelPath: str = "yolov8n.pt"):

        print(f"[INFO] Loading YOLOv8 model: {modelPath}")
        self.model  = YOLO(modelPath)

    # Create Detect Objects Functionality
    def detectObjects(self, frame: np.ndarray, confThresh: float = 0.3) -> List[Dict]:
        
        # Prepare results
        results     = self.model.predict(frame, conf=confThresh, verbose=False)
        detections  = []

    # Loop through each detection result returned by YOLO    
        for result in results:
            for box in result.boxes:
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Extract the confidence score and class index
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                # Get the label name from the model’s class names
                label = self.model.names[cls] if hasattr(self.model, "names") else str(cls)

                # Add the detection info as a dictionary
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "label": label
                })
            
        return detections
