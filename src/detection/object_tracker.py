##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'object_tracker.py'
#
# Description:
# Implements ByteTrack algorithm for object tracking.
# ByteTrack uses both high and low confidence detections
# for more robust tracking compared to SORT.
##############################################

import numpy as np
from typing import List, Dict, Tuple
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    """
    Kalman Filter based tracker for bounding boxes
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize tracker with bounding box [x1, y1, x2, y2]
        """
        # Define constant velocity model (7 states, 4 measurements)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],  # x
            [0,1,0,0,0,1,0],  # y
            [0,0,1,0,0,0,1],  # s (scale/area)
            [0,0,0,1,0,0,0],  # r (aspect ratio)
            [0,0,0,0,1,0,0],  # dx
            [0,0,0,0,0,1,0],  # dy
            [0,0,0,0,0,0,1]   # ds
        ])
        
        # Measurement function
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        # Measurement uncertainty
        self.kf.R[2:,2:] *= 10.
        # Initial state uncertainty
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        # Process uncertainty
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        # Initialize state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # ByteTrack specific
        self.tracklet_len = 0

    def update(self, bbox):
        """Update state with observed bbox"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.tracklet_len += 1
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self):
        """Predict next state and return predicted bbox"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Return current bounding box estimate"""
        return self._convert_x_to_bbox(self.kf.x)

    @staticmethod
    def _convert_bbox_to_z(bbox):
        """
        Convert [x1,y1,x2,y2] to [x,y,s,r]
        where x,y is center, s is scale/area, r is aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h) if h != 0 else 1
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def _convert_x_to_bbox(x, score=None):
        """
        Convert [x,y,s,r] to [x1,y1,x2,y2]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else x[2]
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))


def iou_batch(bboxes1, bboxes2):
    """
    Compute IOU between two sets of bounding boxes
    bboxes: [x1, y1, x2, y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
                + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    
    return iou


def linear_assignment(cost_matrix):
    """Linear assignment using scipy"""
    try:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    except ImportError:
        # Greedy fallback
        matches = []
        cost = cost_matrix.copy()
        for _ in range(min(cost.shape)):
            i, j = np.unravel_index(cost.argmin(), cost.shape)
            if cost[i, j] < 1e10:
                matches.append([i, j])
                cost[i, :] = 1e10
                cost[:, j] = 1e10
        return np.array(matches) if matches else np.empty((0, 2), dtype=int)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assign detections to tracked objects using IOU
    
    Returns:
        matched_indices: array of [detection_idx, tracker_idx] pairs
        unmatched_detections: array of detection indices
        unmatched_trackers: array of tracker indices
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    
    if min(iou_matrix.shape) > 0:
        # Use linear assignment
        matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    
    # Find unmatched detections
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    # Find unmatched trackers
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    # Filter out matches with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    
    Key Innovation:
    - Uses BOTH high and low confidence detections
    - First associates high-confidence detections to tracks
    - Then tries to recover tracks using low-confidence detections
    - Better handling of occlusions and tracking recovery
    
    Reference: https://arxiv.org/abs/2110.06864
    """
    
    def __init__(self, 
                 track_thresh: float = 0.5,      # High confidence threshold
                 track_buffer: int = 30,          # Max frames to keep lost tracks
                 match_thresh: float = 0.8,       # IOU threshold for first association
                 second_match_thresh: float = 0.5 # IOU threshold for second association
                ):
        """
        Initialize ByteTrack
        
        Args:
            track_thresh: Confidence threshold for high-confidence detections
            track_buffer: Number of frames to keep lost tracks alive
            match_thresh: IOU threshold for first association (high conf)
            second_match_thresh: IOU threshold for second association (low conf)
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        
        self.tracked_tracks = []   # Active tracks
        self.lost_tracks = []       # Recently lost tracks
        self.removed_tracks = []    # Permanently removed tracks
        
        self.frame_count = 0
        
        # Store track metadata
        self.track_labels = {}   # track_id -> label
        self.track_info = {}     # track_id -> {first_seen, last_seen, label, confidence}
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with 'bbox', 'label', 'confidence'
            
        Returns:
            List of tracked objects with track IDs
        """
        self.frame_count += 1
        
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []
        
        # Separate high and low confidence detections (ByteTrack innovation!)
        if len(detections) > 0:
            confidences = np.array([d['confidence'] for d in detections])
            
            # High confidence detections
            high_idx = confidences >= self.track_thresh
            high_dets = [d for i, d in enumerate(detections) if high_idx[i]]
            
            # Low confidence detections (for second association)
            low_idx = confidences < self.track_thresh
            low_dets = [d for i, d in enumerate(detections) if low_idx[i]]
        else:
            high_dets = []
            low_dets = []
        
        # Predict all tracks
        for track in self.tracked_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()
        
        # === FIRST ASSOCIATION: High-confidence detections to tracked tracks ===
        if len(high_dets) > 0:
            high_boxes = np.array([d['bbox'] for d in high_dets])
            
            # Get tracked track predictions
            if len(self.tracked_tracks) > 0:
                track_boxes = np.array([t.get_state()[0] for t in self.tracked_tracks])
                
                # Associate
                matches, unmatched_dets, unmatched_tracks = associate_detections_to_trackers(
                    high_boxes, track_boxes, self.match_thresh
                )
                
                # Update matched tracks
                for m in matches:
                    track = self.tracked_tracks[m[1]]
                    det = high_dets[m[0]]
                    track.update(np.array(det['bbox']))
                    activated_tracks.append(track)
                    
                    # Update metadata
                    track_id = track.id
                    self.track_labels[track_id] = det['label']
                    if track_id in self.track_info:
                        self.track_info[track_id]['last_seen'] = self.frame_count
                        self.track_info[track_id]['confidence'] = max(
                            self.track_info[track_id]['confidence'],
                            det['confidence']
                        )
                    else:
                        self.track_info[track_id] = {
                            'first_seen': self.frame_count,
                            'last_seen': self.frame_count,
                            'label': det['label'],
                            'confidence': det['confidence']
                        }
                
                # Handle unmatched detections (create new tracks)
                for i in unmatched_dets:
                    det = high_dets[i]
                    track = KalmanBoxTracker(np.array(det['bbox']))
                    track.update(np.array(det['bbox']))
                    activated_tracks.append(track)
                    
                    track_id = track.id
                    self.track_labels[track_id] = det['label']
                    self.track_info[track_id] = {
                        'first_seen': self.frame_count,
                        'last_seen': self.frame_count,
                        'label': det['label'],
                        'confidence': det['confidence']
                    }
                
                # Mark unmatched tracks as lost
                for i in unmatched_tracks:
                    track = self.tracked_tracks[i]
                    if track.time_since_update <= self.track_buffer:
                        lost_tracks.append(track)
                    else:
                        removed_tracks.append(track)
            else:
                # No existing tracks, create new ones
                for det in high_dets:
                    track = KalmanBoxTracker(np.array(det['bbox']))
                    track.update(np.array(det['bbox']))
                    activated_tracks.append(track)
                    
                    track_id = track.id
                    self.track_labels[track_id] = det['label']
                    self.track_info[track_id] = {
                        'first_seen': self.frame_count,
                        'last_seen': self.frame_count,
                        'label': det['label'],
                        'confidence': det['confidence']
                    }
        
        # === SECOND ASSOCIATION: Low-confidence detections to lost tracks ===
        # This is what makes ByteTrack special!
        if len(low_dets) > 0 and len(lost_tracks) > 0:
            low_boxes = np.array([d['bbox'] for d in low_dets])
            lost_boxes = np.array([t.get_state()[0] for t in lost_tracks])
            
            matches, unmatched_dets, unmatched_tracks = associate_detections_to_trackers(
                low_boxes, lost_boxes, self.second_match_thresh
            )
            
            # Recover lost tracks
            for m in matches:
                track = lost_tracks[m[1]]
                det = low_dets[m[0]]
                track.update(np.array(det['bbox']))
                refind_tracks.append(track)
                
                track_id = track.id
                if track_id in self.track_info:
                    self.track_info[track_id]['last_seen'] = self.frame_count
            
            # Permanently remove tracks that couldn't be recovered
            for i in unmatched_tracks:
                track = lost_tracks[i]
                if track.time_since_update > self.track_buffer:
                    removed_tracks.append(track)
        
        # Update track lists
        self.tracked_tracks = [t for t in activated_tracks + refind_tracks]
        self.lost_tracks = [t for t in lost_tracks if t not in refind_tracks and t not in removed_tracks]
        self.removed_tracks.extend(removed_tracks)
        
        # Return active tracked objects
        output_tracks = []
        for track in self.tracked_tracks:
            bbox = track.get_state()[0]
            track_id = track.id
            
            output_tracks.append({
                'track_id': track_id,
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                'label': self.track_labels.get(track_id, 'unknown'),
                'confidence': self.track_info.get(track_id, {}).get('confidence', 0.0),
                'hits': track.hits,
                'age': track.age,
                'tracklet_len': track.tracklet_len
            })
        
        return output_tracks
    
    def get_unique_items(self) -> Dict[int, Dict]:
        """Get all tracked items"""
        return self.track_info


# Alias for compatibility
ObjectTracker = ByteTracker