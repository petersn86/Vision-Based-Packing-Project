##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'entry_detector.py'
#
# Description:
# Detects when items are inside box regions by tracking
# their spatial overlap over time. Works for both items
# entering boxes AND items already in boxes.
#
##############################################

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

class EntryDetector:
    """
    Tracks object positions to detect when items are inside boxes.
    Can detect both items entering boxes AND items already in boxes.
    """
    
    def __init__(self, 
                 entry_threshold: int = 3,  # Frames to confirm entry
                 exit_threshold: int = 5,   # Frames before considering exit
                 require_motion: bool = False):  # NEW: whether to require outside→inside motion
        """
        Initialize entry detector
        
        Args:
            entry_threshold: Number of consecutive frames inside box to confirm entry
            exit_threshold: Number of frames outside before considering item removed
            require_motion: If True, only detect items that move INTO boxes.
                           If False, also detect items already in boxes.
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.require_motion = require_motion
        
        # Track object positions over time
        self.track_history = defaultdict(list)  # track_id -> [(frame, bbox, box_id)]
        
        # Track which items have entered which boxes
        self.entered_items = {}  # track_id -> box_id
        self.confirmed_entries = set()  # (track_id, box_id) tuples
        
        # Track consecutive frames inside/outside box
        self.inside_counter = defaultdict(int)  # (track_id, box_id) -> frame_count
        self.outside_counter = defaultdict(int)  # (track_id, box_id) -> frame_count
    
    def get_item_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def is_point_in_box(self, point: Tuple[int, int], box_bbox: List[int]) -> bool:
        """Check if point is inside box bounding box"""
        x, y = point
        x1, y1, x2, y2 = box_bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def get_overlap_percentage(self, item_bbox: List[int], box_bbox: List[int]) -> float:
        """
        Calculate what percentage of the item overlaps with the box
        Returns value between 0.0 and 1.0
        """
        ix1, iy1, ix2, iy2 = item_bbox
        bx1, by1, bx2, by2 = box_bbox
        
        # Calculate intersection
        x1 = max(ix1, bx1)
        y1 = max(iy1, by1)
        x2 = min(ix2, bx2)
        y2 = min(iy2, by2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        item_area = (ix2 - ix1) * (iy2 - iy1)
        
        if item_area == 0:
            return 0.0
        
        return intersection_area / item_area
    
    def detect_entry(self, 
                     track_id: int,
                     item_bbox: List[int],
                     box_id: str,
                     box_bbox: List[int],
                     frame_number: int,
                     overlap_threshold: float = 0.5) -> Optional[str]:
        """
        Detect if an item is inside a box
        
        Args:
            track_id: Object tracking ID
            item_bbox: Item bounding box [x1, y1, x2, y2]
            box_id: Box identifier
            box_bbox: Box bounding box [x1, y1, x2, y2]
            frame_number: Current frame number
            overlap_threshold: Minimum overlap percentage to consider "inside"
            
        Returns:
            Box ID if entry is confirmed, None otherwise
        """
        key = (track_id, box_id)
        
        # Calculate overlap
        overlap = self.get_overlap_percentage(item_bbox, box_bbox)
        is_inside = overlap >= overlap_threshold
        
        # Update history
        self.track_history[track_id].append({
            'frame': frame_number,
            'bbox': item_bbox,
            'box_id': box_id,
            'overlap': overlap,
            'inside': is_inside
        })
        
        # Keep only recent history (last 30 frames)
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id] = self.track_history[track_id][-30:]
        
        # If already confirmed entry, don't re-confirm
        if key in self.confirmed_entries:
            # Check if item has exited
            if not is_inside:
                self.outside_counter[key] += 1
                if self.outside_counter[key] >= self.exit_threshold:
                    # Item has left the box
                    self.confirmed_entries.discard(key)
                    self.inside_counter[key] = 0
                    self.outside_counter[key] = 0
            else:
                self.outside_counter[key] = 0
            return None
        
        # Track consecutive frames inside box
        if is_inside:
            self.inside_counter[key] += 1
            self.outside_counter[key] = 0
            
            # Check if entry is confirmed
            if self.inside_counter[key] >= self.entry_threshold:
                # MODIFIED: Only verify motion if required
                if self.require_motion:
                    # Strict mode: item must have moved INTO box
                    if self._verify_entry_motion(track_id, box_id, box_bbox):
                        self.confirmed_entries.add(key)
                        self.entered_items[track_id] = box_id
                        return box_id
                else:
                    # Relaxed mode: accept any item that stays in box long enough
                    self.confirmed_entries.add(key)
                    self.entered_items[track_id] = box_id
                    return box_id
        else:
            self.inside_counter[key] = 0
            self.outside_counter[key] += 1
        
        return None
    
    def _verify_entry_motion(self, 
                             track_id: int, 
                             box_id: str,
                             box_bbox: List[int]) -> bool:
        """
        Verify that the item actually moved INTO the box
        (not just detected inside from the start)
        
        Only used when require_motion=True
        """
        history = self.track_history[track_id]
        
        if len(history) < self.entry_threshold + 2:
            return False
        
        # Check if item was outside box in recent frames
        recent_history = history[-(self.entry_threshold + 5):]
        
        was_outside = False
        now_inside = False
        
        for record in recent_history[:len(recent_history)//2]:
            if record['box_id'] == box_id and not record['inside']:
                was_outside = True
        
        for record in recent_history[len(recent_history)//2:]:
            if record['box_id'] == box_id and record['inside']:
                now_inside = True
        
        # Entry is valid if item transitioned from outside to inside
        return was_outside and now_inside
    
    def get_box_for_item(self, track_id: int) -> Optional[str]:
        """Get the box ID that an item has entered"""
        return self.entered_items.get(track_id)
    
    def get_all_entries(self) -> Dict[str, List[Tuple[int, str]]]:
        """
        Get all confirmed entries grouped by box
        
        Returns:
            Dict mapping box_id to list of (track_id, label) tuples
        """
        entries_by_box = defaultdict(list)
        
        for (track_id, box_id) in self.confirmed_entries:
            entries_by_box[box_id].append(track_id)
        
        return dict(entries_by_box)
    
    def reset(self):
        """Reset all tracking data"""
        self.track_history.clear()
        self.entered_items.clear()
        self.confirmed_entries.clear()
        self.inside_counter.clear()
        self.outside_counter.clear()