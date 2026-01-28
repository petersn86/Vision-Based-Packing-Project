##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'qr_detector.py'
#
# Description:
# Detects and decodes QR codes in video frames to identify
# boxes/containers. Maps detected items to boxes based on
# spatial proximity.
#
##############################################

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pyzbar import pyzbar

class QRDetector:
    """
    Detects QR codes and maps items to boxes based on spatial proximity
    """
    
    def __init__(self, proximity_threshold: int = 100):
        """
        Initialize QR detector
        
        Args:
            proximity_threshold: Distance threshold (pixels) for item-to-box mapping
        """
        self.proximity_threshold = proximity_threshold
        self.detected_boxes = {}  # box_id -> {bbox, last_seen_frame}
        
    def detect_qr_codes(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect QR codes in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detected QR codes with data and bounding boxes
        """
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect QR codes
        qr_codes = pyzbar.decode(gray)
        
        results = []
        for qr in qr_codes:
            # Extract data
            data = qr.data.decode('utf-8')
            qr_type = qr.type
            
            # Get bounding box
            x, y, w, h = qr.rect
            bbox = [x, y, x + w, y + h]
            
            # Get polygon points for more precise boundary
            points = [(point.x, point.y) for point in qr.polygon]
            
            results.append({
                'data': data,
                'type': qr_type,
                'bbox': bbox,
                'polygon': points,
                'center': (x + w // 2, y + h // 2)
            })
        
        return results
    
    def update_boxes(self, qr_codes: List[Dict], frame_index: int):
        """
        Update detected boxes registry
        
        Args:
            qr_codes: List of detected QR codes
            frame_index: Current frame number
        """
        for qr in qr_codes:
            box_id = qr['data']
            self.detected_boxes[box_id] = {
                'bbox': qr['bbox'],
                'center': qr['center'],
                'last_seen_frame': frame_index
            }
    
    def get_active_boxes(self, current_frame: int, max_age: int = 30) -> Dict:
        """
        Get boxes that were recently detected
        
        Args:
            current_frame: Current frame number
            max_age: Maximum frames since last detection
            
        Returns:
            Dictionary of active boxes
        """
        active = {}
        for box_id, info in self.detected_boxes.items():
            if current_frame - info['last_seen_frame'] <= max_age:
                active[box_id] = info
        return active
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def map_item_to_box(self, 
                        item_bbox: List[int], 
                        current_frame: int,
                        max_box_age: int = 30) -> Optional[str]:
        """
        Map an item to the nearest box based on spatial proximity
        
        Args:
            item_bbox: Item bounding box [x1, y1, x2, y2]
            current_frame: Current frame number
            max_box_age: Maximum age for considering a box
            
        Returns:
            Box ID if mapped, None otherwise
        """
        # Calculate item center
        item_center = (
            (item_bbox[0] + item_bbox[2]) // 2,
            (item_bbox[1] + item_bbox[3]) // 2
        )
        
        # Get active boxes
        active_boxes = self.get_active_boxes(current_frame, max_box_age)
        
        if not active_boxes:
            return None
        
        # Find nearest box
        nearest_box = None
        min_distance = float('inf')
        
        for box_id, box_info in active_boxes.items():
            box_center = box_info['center']
            distance = self.calculate_distance(item_center, box_center)
            
            if distance < min_distance and distance < self.proximity_threshold:
                min_distance = distance
                nearest_box = box_id
        
        return nearest_box
    
    def check_item_inside_box(self, item_bbox: List[int], box_bbox: List[int]) -> bool:
        """
        Check if item center is inside box bounding box
        
        Args:
            item_bbox: Item bounding box [x1, y1, x2, y2]
            box_bbox: Box bounding box [x1, y1, x2, y2]
            
        Returns:
            True if item is inside box
        """
        # Calculate item center
        item_center_x = (item_bbox[0] + item_bbox[2]) // 2
        item_center_y = (item_bbox[1] + item_bbox[3]) // 2
        
        # Check if center is inside box
        return (box_bbox[0] <= item_center_x <= box_bbox[2] and
                box_bbox[1] <= item_center_y <= box_bbox[3])
    
    def annotate_qr_codes(self, frame: np.ndarray, qr_codes: List[Dict]) -> np.ndarray:
        """
        Draw QR codes on frame for visualization
        
        Args:
            frame: Input frame
            qr_codes: List of detected QR codes
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for qr in qr_codes:
            # Draw bounding box
            bbox = qr['bbox']
            cv2.rectangle(
                annotated,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 0, 255),  # Red for QR codes
                3
            )
            
            # Draw polygon outline
            if 'polygon' in qr:
                points = np.array(qr['polygon'], dtype=np.int32)
                cv2.polylines(annotated, [points], True, (0, 255, 255), 2)
            
            # Draw label
            text = f"Box: {qr['data']}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            cv2.rectangle(
                annotated,
                (bbox[0], bbox[1] - text_height - baseline - 10),
                (bbox[0] + text_width, bbox[1]),
                (0, 0, 255),
                -1
            )
            
            cv2.putText(
                annotated,
                text,
                (bbox[0], bbox[1] - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return annotated


class BoxItemMapper:
    """
    Maintains mappings between boxes and items over time
    """
    
    def __init__(self):
        """Initialize mapper"""
        self.box_items = {}  # box_id -> set of (track_id, label, timestamp)
        self.item_boxes = {}  # track_id -> (box_id, timestamp)
    
    def add_mapping(self, 
                    track_id: int, 
                    label: str, 
                    box_id: str,
                    timestamp: float):
        """
        Add item-to-box mapping
        
        Args:
            track_id: Object tracking ID
            label: Item label
            box_id: Box identifier
            timestamp: Detection timestamp
        """
        # Add to box_items
        if box_id not in self.box_items:
            self.box_items[box_id] = set()
        
        self.box_items[box_id].add((track_id, label, timestamp))
        
        # Add to item_boxes
        self.item_boxes[track_id] = (box_id, timestamp)
    
    def get_box_contents(self, box_id: str) -> List[Tuple]:
        """Get all items in a box"""
        return sorted(list(self.box_items.get(box_id, set())))
    
    def get_item_box(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Get box assignment for an item"""
        return self.item_boxes.get(track_id)
    
    def get_all_boxes(self) -> Dict[str, List]:
        """Get all boxes with their contents"""
        result = {}
        for box_id, items in self.box_items.items():
            result[box_id] = sorted(list(items), key=lambda x: x[2])  # Sort by timestamp
        return result
    
    def export_to_dict(self) -> Dict:
        """Export mappings as dictionary"""
        result = {
            'boxes': {},
            'summary': {
                'total_boxes': len(self.box_items),
                'total_items': len(self.item_boxes)
            }
        }
        
        for box_id, items in self.box_items.items():
            result['boxes'][box_id] = [
                {
                    'track_id': track_id,
                    'label': label,
                    'timestamp': timestamp
                }
                for track_id, label, timestamp in sorted(items, key=lambda x: x[2])
            ]
        
        return result
