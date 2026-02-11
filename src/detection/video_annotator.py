##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'video_annotator.py'
#
# Description:
# Creates annotated video output with bounding boxes,
# labels, confidence scores, and tracking IDs overlaid
# on the original video frames. Optionally displays a
# hand-detected indicator banner when hand_detected=True.
#
##############################################

import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

class VideoAnnotator:
    """
    Annotates video frames with detection and tracking information
    """
    
    def __init__(self, output_path: str, fps: int = 10, codec: str = 'mp4v'):
        """
        Initialize video annotator
        
        Args:
            output_path: Path to save annotated video
            fps: Frames per second for output video
            codec: Video codec (mp4v, x264, etc.)
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.frame_size = None
        
        # Color palette for different track IDs
        self.colors = self._generate_colors(100)
    
    def _generate_colors(self, n: int) -> List[tuple]:
        """Generate n distinct colors for visualization"""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), 
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors
    
    def _init_writer(self, frame_shape):
        """Initialize video writer with frame dimensions"""
        height, width = frame_shape[:2]
        self.frame_size = (width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.frame_size
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {self.output_path}")
    
    def annotate_frame(self, 
                       frame: np.ndarray, 
                       detections: List[Dict],
                       frame_number: Optional[int] = None,
                       timestamp: Optional[float] = None,
                       hand_detected: bool = False) -> np.ndarray:
        """
        Annotate a single frame with detections
        
        Args:
            frame: Input frame (BGR format)
            detections: List of detection dicts with bbox, label, confidence, track_id
            frame_number: Optional frame number to display
            timestamp: Optional timestamp to display
            hand_detected: If True, draw a hand-detected banner on the frame
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw each detection
        for det in detections:
            bbox = det['bbox']
            label = det.get('label', 'unknown')
            confidence = det.get('confidence', 0.0)
            track_id = det.get('track_id', None)
            
            # Get color based on track ID
            if track_id is not None:
                color = self.colors[track_id % len(self.colors)]
            else:
                color = (0, 255, 0)  # Green for untracked
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            if track_id is not None:
                text = f"ID{track_id}: {label} ({confidence:.2f})"
            else:
                text = f"{label} ({confidence:.2f})"
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Draw frame info overlay
        info_text = []
        if frame_number is not None:
            info_text.append(f"Frame: {frame_number}")
        if timestamp is not None:
            info_text.append(f"Time: {timestamp:.2f}s")
        info_text.append(f"Detections: {len(detections)}")
        
        # Draw info box in top-left corner
        y_offset = 30
        for text in info_text:
            cv2.putText(
                annotated,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.putText(
                annotated,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
            y_offset += 25

        # Draw hand detected banner at the bottom of the frame
        if hand_detected:
            h, w = annotated.shape[:2]
            banner_height = 40
            banner_y = h - banner_height

            # Semi-transparent red background
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, banner_y), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            # Banner text
            banner_text = "HAND DETECTED"
            (tw, th), bl = cv2.getTextSize(
                banner_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2
            )
            text_x = (w - tw) // 2
            text_y = banner_y + (banner_height + th) // 2

            cv2.putText(
                annotated,
                banner_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    def add_frame(self, 
                  frame: np.ndarray, 
                  detections: List[Dict],
                  frame_number: Optional[int] = None,
                  timestamp: Optional[float] = None,
                  hand_detected: bool = False):
        """
        Add annotated frame to output video
        
        Args:
            frame: Input frame
            detections: List of detections
            frame_number: Optional frame number
            timestamp: Optional timestamp
            hand_detected: If True, draw hand-detected banner
        """
        # Initialize writer on first frame
        if self.writer is None:
            self._init_writer(frame.shape)
        
        # Annotate and write frame
        annotated = self.annotate_frame(
            frame, detections, frame_number, timestamp, hand_detected
        )
        self.writer.write(annotated)
    
    def finalize(self):
        """Close video writer and finalize output"""
        if self.writer is not None:
            self.writer.release()
            print(f"[INFO] Annotated video saved to: {self.output_path}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finalize()


def create_annotated_video(frames_data: List[Dict],
                          detections_per_frame: Dict[int, List[Dict]],
                          output_path: str,
                          fps: int = 10,
                          codec: str = 'mp4v') -> str:
    """
    Convenience function to create annotated video from frame data
    
    Args:
        frames_data: List of frame dicts with 'frame', 'index', 'timestamp'
        detections_per_frame: Dict mapping frame_index to list of detections
        output_path: Output video path
        fps: Output video FPS
        codec: Video codec
        
    Returns:
        Path to created video
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with VideoAnnotator(output_path, fps, codec) as annotator:
        for frame_data in frames_data:
            frame_idx = frame_data['index']
            frame = frame_data['frame']
            timestamp = frame_data.get('timestamp')
            
            detections = detections_per_frame.get(frame_idx, [])
            annotator.add_frame(frame, detections, frame_idx, timestamp)
    
    return output_path