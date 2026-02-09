##############################################
# @Author: Evan Rapoza
# @Contributor(s): Peter Nolan
# @Document: 'object_detector.py'
#
# Description:
# This module performs object detection on video frames using
# a pretrained YOLO model. Using object orientated design,
# it returns bounding boxes, labels, and confidence
# scores for each detected object. Supports separate confidence
# thresholds for primary model and box detection model.
# 
# Features:
# - Dual model support (general + box-specific)
# - Test-Time Augmentation (TTA) for improved detection
# - Configurable confidence thresholds
# - Support for YOLO11 and YOLO26 models
#
##############################################

import numpy as np
import cv2
from ultralytics import YOLO
from typing import List, Dict, Optional

class ObjectDetector:

    def __init__(self, 
                 modelPath: str = "yolo26m.pt", 
                 boxModelPath: Optional[str] = None,
                 useTTA: bool = False):
        """
        Initialize Object Detector with YOLO models
        
        Args:
            modelPath: Path to primary YOLO model for general object detection
            boxModelPath: Path to secondary YOLO model for box/carton detection (optional)
            useTTA: Enable Test-Time Augmentation for improved detection accuracy
        """
        
        print(f"[INFO] Loading YOLO model: {modelPath}")
        self.model = YOLO(modelPath)
        self.useTTA = useTTA
        
        if useTTA:
            print(f"[INFO] Test-Time Augmentation (TTA) ENABLED - slower but more accurate")
        
        # Optional second model for detecting boxes
        self.box_model = None
        if boxModelPath:
            print(f"[INFO] Loading secondary YOLO model (box detector): {boxModelPath}")
            self.box_model = YOLO(boxModelPath)

    def detectObjects(self, 
                      frame: np.ndarray, 
                      confThresh: float = 0.35, 
                      boxConfThresh: Optional[float] = None,
                      useTTA: Optional[bool] = None) -> List[Dict]:
        """
        Detect objects in frame using primary and optional box models
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            confThresh: Confidence threshold for primary model (0.0 - 1.0)
            boxConfThresh: Confidence threshold for box model (uses confThresh if None)
            useTTA: Override instance TTA setting for this call (None uses instance default)
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score
                - label: Class label string
        """
        
        # Use same threshold for boxes if not specified
        if boxConfThresh is None:
            boxConfThresh = confThresh
        
        # Determine if TTA should be used for this call
        augment = useTTA if useTTA is not None else self.useTTA
        
        # Run primary model prediction
        results = self.model.predict(
            frame, 
            conf=confThresh, 
            verbose=False,
            augment=augment
        )
        
        detections = []

        # Process each detection result from primary model
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Extract confidence score and class index
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                # Get label name from model's class names
                label = self.model.names[cls] if hasattr(self.model, "names") else str(cls)

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "label": label
                })

        # Run secondary (box) model if available
        if self.box_model:
            box_results = self.box_model.predict(
                frame, 
                conf=boxConfThresh, 
                verbose=False,
                augment=augment
            )
            
            for result in box_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = self.box_model.names[cls] if hasattr(self.box_model, "names") else str(cls)

                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": conf,
                        "label": label
                    })
            
        return detections
    
    def detectWithMultiScale(self,
                             frame: np.ndarray,
                             confThresh: float = 0.35,
                             scales: List[float] = [0.75, 1.0, 1.25]) -> List[Dict]:
        """
        Run detection at multiple scales and merge results.
        Useful for detecting objects at varying distances/sizes.
        
        Args:
            frame: Input frame
            confThresh: Confidence threshold
            scales: List of scale factors to try
            
        Returns:
            Merged list of detections (with NMS applied)
        """
        all_detections = []
        original_h, original_w = frame.shape[:2]
        
        for scale in scales:
            # Resize frame
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            scaled_frame = cv2.resize(frame, (new_w, new_h))
            
            # Detect on scaled frame
            detections = self.detectObjects(scaled_frame, confThresh, useTTA=False)
            
            # Scale bounding boxes back to original size
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                det['bbox'] = [
                    int(x1 / scale),
                    int(y1 / scale),
                    int(x2 / scale),
                    int(y2 / scale)
                ]
                all_detections.append(det)
        
        # Apply simple NMS to remove duplicates
        return self._apply_nms(all_detections, iou_threshold=0.5)
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IOU threshold for considering boxes as duplicates
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Group by label
        label_groups = {}
        for det in detections:
            label = det['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(det)
        
        final_detections = []
        
        for label, dets in label_groups.items():
            # Sort by confidence (highest first)
            dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)
                
                # Remove overlapping boxes
                dets = [d for d in dets if self._calculate_iou(best['bbox'], d['bbox']) < iou_threshold]
            
            final_detections.extend(keep)
        
        return final_detections
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def getModelInfo(self) -> Dict:
        """Get information about loaded models"""
        info = {
            "primary_model": {
                "path": str(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else "unknown",
                "classes": len(self.model.names) if hasattr(self.model, 'names') else 0,
                "class_names": list(self.model.names.values()) if hasattr(self.model, 'names') else []
            },
            "tta_enabled": self.useTTA
        }
        
        if self.box_model:
            info["box_model"] = {
                "path": str(self.box_model.ckpt_path) if hasattr(self.box_model, 'ckpt_path') else "unknown",
                "classes": len(self.box_model.names) if hasattr(self.box_model, 'names') else 0,
                "class_names": list(self.box_model.names.values()) if hasattr(self.box_model, 'names') else []
            }
        
        return info