##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'object_tracker.py'
#
# Description:
# Implements ByteTrack algorithm for object tracking with
# enhanced re-identification using deep visual embeddings
# (MobileNetV2). This produces far more discriminative
# features than color histograms, allowing objects to
# regain their original track ID even after large positional
# jumps or partial occlusion (e.g. being held by a hand).
##############################################

import numpy as np
from typing import List, Dict, Tuple, Optional
from filterpy.kalman import KalmanFilter
from collections import defaultdict
import cv2

# ---------------------------------------------------------------------------
# Deep feature extractor using MobileNetV2
# ---------------------------------------------------------------------------
# We use OpenCV's DNN module to run MobileNetV2 — no PyTorch/TF dependency.
# The model is downloaded once from opencv's model zoo and cached locally.
# Falls back to HSV histograms if the model cannot be loaded.
# ---------------------------------------------------------------------------

import os
import urllib.request

# MobileNetV2 SSD feature extraction config
_MOBILENET_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_MOBILENET_MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224.tflite"

# We'll use a simpler approach: MobileNetV2 via torchvision if available,
# otherwise graceful fallback to enhanced histogram
_TORCH_AVAILABLE = False
try:
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class DeepFeatureExtractor:
    """
    Deep visual feature extractor using MobileNetV2.

    Produces 1280-dim L2-normalised embeddings from image crops.
    These are far more discriminative than color histograms —
    they capture shape, texture, and object-level semantics,
    so a Dial soap bottle looks different from a water bottle
    even if they have similar color distributions.

    Falls back to an enhanced multi-channel histogram if PyTorch
    is not available (still better than the original HSV-only approach).
    """

    def __init__(self):
        self.use_deep = False
        self.model = None
        self.transform = None

        if _TORCH_AVAILABLE:
            try:
                self._init_mobilenet()
                self.use_deep = True
                print("[INFO] FeatureExtractor: using MobileNetV2 deep embeddings")
            except Exception as e:
                print(f"[WARNING] MobileNetV2 init failed ({e}), falling back to histogram")
        else:
            print("[INFO] FeatureExtractor: PyTorch not available, using enhanced histogram")

    def _init_mobilenet(self):
        """Load MobileNetV2, strip the classifier, keep the feature backbone."""
        weights = tv_models.MobileNet_V2_Weights.DEFAULT
        backbone = tv_models.mobilenet_v2(weights=weights)
        # Remove the final classifier — keep everything up to the avg pool
        # Output: (1, 1280, 1, 1) → flattened to (1280,)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval()

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract feature vector from an image crop.

        Args:
            frame: Full BGR frame
            bbox:  [x1, y1, x2, y2]

        Returns:
            1D numpy feature vector (L2-normalised), or None on failure
        """
        crop = self._safe_crop(frame, bbox)
        if crop is None:
            return None

        if self.use_deep:
            return self._deep_embed(crop)
        else:
            return self._histogram_embed(crop)

    def _safe_crop(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Safely crop frame, clamping to image bounds."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _deep_embed(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run MobileNetV2 forward pass, return L2-normalised embedding."""
        try:
            # OpenCV uses BGR; torchvision expects RGB
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.model(tensor)           # (1, 1280, 1, 1)
                feat = feat.squeeze().cpu().numpy() # (1280,)

            # L2 normalise
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            return feat.astype(np.float32)
        except Exception as e:
            return None

    def _histogram_embed(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Enhanced fallback: multi-channel histogram (H, S, V + R, G, B).
        Better than original HSV-only approach.
        """
        bins = 32
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hists = []
        for img, channels in [(hsv, [0, 1, 2]), (crop_bgr, [0, 1, 2])]:
            for ch in channels:
                rng = [0, 180] if (img is hsv and ch == 0) else [0, 256]
                h = cv2.calcHist([img], [ch], None, [bins], rng)
                h = cv2.normalize(h, h).flatten()
                hists.append(h)
        feat = np.concatenate(hists)
        norm = np.linalg.norm(feat)
        return (feat / norm).astype(np.float32) if norm > 0 else feat

    def compute_similarity(self,
                            feat1: Optional[np.ndarray],
                            feat2: Optional[np.ndarray]) -> float:
        """Cosine similarity between two feature vectors (0–1)."""
        if feat1 is None or feat2 is None:
            return 0.0
        n1, n2 = np.linalg.norm(feat1), np.linalg.norm(feat2)
        if n1 == 0 or n2 == 0:
            return 0.0
        # Both vectors should already be L2-normalised, but clip for safety
        return float(np.clip(np.dot(feat1, feat2) / (n1 * n2), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Kalman tracker (unchanged from original)
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """Kalman Filter based tracker for bounding boxes."""
    count = 0

    def __init__(self, bbox, feature=None):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.tracklet_len = 0

        self.feature = feature
        self.feature_history: List[np.ndarray] = []
        if feature is not None:
            self.feature_history.append(feature)

    def update(self, bbox, feature=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.tracklet_len += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        if feature is not None:
            self.feature = feature
            self.feature_history.append(feature)
            if len(self.feature_history) > 10:
                self.feature_history = self.feature_history[-10:]

    def predict(self):
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
        return self._convert_x_to_bbox(self.kf.x)

    def get_average_feature(self):
        if not self.feature_history:
            return None
        return np.mean(self.feature_history, axis=0)

    @staticmethod
    def _convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h != 0 else 1
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def _convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else x[2]
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# ---------------------------------------------------------------------------
# Association utilities (unchanged)
# ---------------------------------------------------------------------------

def iou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2]-bboxes1[..., 0]) * (bboxes1[..., 3]-bboxes1[..., 1])
                + (bboxes2[..., 2]-bboxes2[..., 0]) * (bboxes2[..., 3]-bboxes2[..., 1]) - wh)
    return iou


def linear_assignment(cost_matrix):
    try:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    except ImportError:
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
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=int)

    # Use explicit length checks — never rely on numpy array truthiness
    if matched_indices.shape[0] > 0:
        matched_det_idx   = set(matched_indices[:, 0].tolist())
        matched_track_idx = set(matched_indices[:, 1].tolist())
    else:
        matched_det_idx   = set()
        matched_track_idx = set()

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_det_idx]
    unmatched_trackers   = [t for t in range(len(trackers))   if t not in matched_track_idx]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(int(m[0]))
            unmatched_trackers.append(int(m[1]))
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ---------------------------------------------------------------------------
# Appearance-only association
# ---------------------------------------------------------------------------

def associate_by_appearance(detections: List[Dict],
                             gallery_tracks: List[KalmanBoxTracker],
                             feature_extractor: DeepFeatureExtractor,
                             frame: np.ndarray,
                             similarity_threshold: float,
                             label_map: Dict[int, str]) -> Tuple[List, List, List]:
    """
    Match unmatched detections to gallery tracks purely by visual similarity.

    Returns:
        matched:   list of (det_idx, track) pairs
        unmatched: list of det indices that had no gallery match
        used_tracks: set of matched track objects (to remove from gallery)
    """
    if not gallery_tracks or not detections or frame is None:
        return [], list(range(len(detections))), []

    matched = []
    unmatched = list(range(len(detections)))
    used_tracks = []

    # Build similarity matrix: detections × gallery tracks
    det_features = []
    for det in detections:
        feat = feature_extractor.extract(frame, det['bbox'])
        det_features.append(feat)

    # Greedy matching: best similarity first
    sim_matrix = np.zeros((len(detections), len(gallery_tracks)))
    for di, feat in enumerate(det_features):
        for ti, track in enumerate(gallery_tracks):
            # Only match same label
            if label_map.get(track.id, '') != detections[di]['label']:
                continue
            track_feat = track.get_average_feature()
            sim_matrix[di, ti] = feature_extractor.compute_similarity(feat, track_feat)

    # Assign highest-similarity pairs above threshold
    assigned_dets = set()
    assigned_tracks = set()

    # Flatten and sort by similarity descending
    pairs = sorted(
        [(sim_matrix[di, ti], di, ti)
         for di in range(len(detections))
         for ti in range(len(gallery_tracks))],
        reverse=True
    )

    for sim, di, ti in pairs:
        if sim < similarity_threshold:
            break
        if di in assigned_dets or ti in assigned_tracks:
            continue
        matched.append((di, gallery_tracks[ti]))
        assigned_dets.add(di)
        assigned_tracks.add(ti)
        used_tracks.append(gallery_tracks[ti])

    unmatched = [di for di in range(len(detections)) if di not in assigned_dets]
    return matched, unmatched, used_tracks


# ---------------------------------------------------------------------------
# ByteTracker with deep re-ID
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    ByteTrack with deep visual re-identification.

    Key change from original: FeatureExtractor replaced with
    DeepFeatureExtractor (MobileNetV2). The re-ID gallery association
    step now uses a dedicated appearance-only matcher, so objects that
    jump position between frames (e.g. held items) can still be
    re-identified by what they look like rather than where they are.

    Association priority per frame:
      1. IOU matching  (high-conf detections ↔ active tracks)
      2. IOU matching  (low-conf detections  ↔ lost tracks)     [ByteTrack step 2]
      3. Appearance matching (unmatched high-conf ↔ gallery)    [deep re-ID]
      4. New track creation (truly new objects)
    """

    def __init__(self,
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 second_match_thresh: float = 0.5,
                 reid_enabled: bool = True,
                 reid_thresh: float = 0.5,
                 reid_buffer: int = 90):

        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.reid_enabled = reid_enabled
        self.reid_thresh = reid_thresh
        self.reid_buffer = reid_buffer

        self.tracked_tracks: List[KalmanBoxTracker] = []
        self.lost_tracks: List[KalmanBoxTracker] = []
        self.removed_tracks: List[KalmanBoxTracker] = []
        self.reid_gallery: List[KalmanBoxTracker] = []

        self.frame_count = 0
        self.current_frame = None

        # Deep feature extractor (MobileNetV2 → histogram fallback)
        self.feature_extractor = DeepFeatureExtractor() if reid_enabled else None

        self.track_labels: Dict[int, str] = {}
        self.track_info: Dict[int, Dict] = {}

    # ------------------------------------------------------------------

    def update(self, detections: List[Dict], frame: np.ndarray = None) -> List[Dict]:
        self.frame_count += 1
        self.current_frame = frame

        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []

        # Split by confidence
        if detections:
            confs = np.array([d['confidence'] for d in detections])
            high_dets = [d for d, m in zip(detections, confs >= self.track_thresh) if m]
            low_dets  = [d for d, m in zip(detections, confs < self.track_thresh)  if m]
        else:
            high_dets, low_dets = [], []

        # Predict all tracks forward
        for t in self.tracked_tracks + self.lost_tracks:
            t.predict()

        # === STEP 1: IOU — high-conf detections ↔ active tracks ===
        unmatched_high: List[int] = list(range(len(high_dets)))

        if high_dets and self.tracked_tracks:
            high_boxes  = np.array([d['bbox'] for d in high_dets])
            track_boxes = np.array([t.get_state()[0] for t in self.tracked_tracks])
            matches, unmatched_high_arr, unmatched_track_idx_arr = associate_detections_to_trackers(
                high_boxes, track_boxes, self.match_thresh
            )
            # Convert to plain Python lists immediately — avoids numpy boolean ambiguity
            unmatched_high       = unmatched_high_arr.tolist()
            unmatched_track_idx  = unmatched_track_idx_arr.tolist()
            for m in matches:
                track = self.tracked_tracks[m[1]]
                det   = high_dets[m[0]]
                feat  = self.feature_extractor.extract(frame, det['bbox']) if self.reid_enabled and frame is not None else None
                track.update(np.array(det['bbox']), feat)
                activated_tracks.append(track)
                self._update_track_info(track.id, det)

            # Mark unmatched active tracks as lost
            for i in unmatched_track_idx:
                t = self.tracked_tracks[i]
                if t.time_since_update <= self.track_buffer:
                    lost_tracks.append(t)
                else:
                    if self.reid_enabled:
                        self.reid_gallery.append(t)
                    removed_tracks.append(t)
        elif self.tracked_tracks:
            # All active tracks lost this frame
            for t in self.tracked_tracks:
                if t.time_since_update <= self.track_buffer:
                    lost_tracks.append(t)
                else:
                    if self.reid_enabled:
                        self.reid_gallery.append(t)
                    removed_tracks.append(t)

        # === STEP 2: IOU — low-conf detections ↔ lost tracks ===
        if low_dets and lost_tracks:
            low_boxes  = np.array([d['bbox'] for d in low_dets])
            lost_boxes = np.array([t.get_state()[0] for t in lost_tracks])
            matches, _, unmatched_lost_idx_arr = associate_detections_to_trackers(
                low_boxes, lost_boxes, self.second_match_thresh
            )
            unmatched_lost_idx = unmatched_lost_idx_arr.tolist()
            for m in matches:
                track = lost_tracks[m[1]]
                det   = low_dets[m[0]]
                feat  = self.feature_extractor.extract(frame, det['bbox']) if self.reid_enabled and frame is not None else None
                track.update(np.array(det['bbox']), feat)
                refind_tracks.append(track)
                self._update_track_info(track.id, det)

            for i in unmatched_lost_idx:
                t = lost_tracks[i]
                if t.time_since_update > self.track_buffer:
                    if self.reid_enabled:
                        self.reid_gallery.append(t)
                    removed_tracks.append(t)

        # === STEP 3: Appearance-only — unmatched high-conf ↔ gallery ===
        truly_new = []
        if self.reid_enabled and len(unmatched_high) > 0 and self.reid_gallery and frame is not None:
            unmatched_dets = [high_dets[i] for i in unmatched_high]
            app_matched, app_unmatched, used = associate_by_appearance(
                unmatched_dets,
                self.reid_gallery,
                self.feature_extractor,
                frame,
                self.reid_thresh,
                self.track_labels
            )
            for det_local_idx, track in app_matched:
                det  = unmatched_dets[det_local_idx]
                feat = self.feature_extractor.extract(frame, det['bbox'])
                track.update(np.array(det['bbox']), feat)
                track.time_since_update = 0
                refind_tracks.append(track)
                self._update_track_info(track.id, det)
                if track in self.reid_gallery:
                    self.reid_gallery.remove(track)

            truly_new = [unmatched_high[i] for i in app_unmatched]
        else:
            truly_new = list(unmatched_high)

        # === STEP 4: Create new tracks for unmatched detections ===
        for i in truly_new:
            det  = high_dets[i]
            feat = self.feature_extractor.extract(frame, det['bbox']) if self.reid_enabled and frame is not None else None
            track = KalmanBoxTracker(np.array(det['bbox']), feat)
            track.update(np.array(det['bbox']), feat)
            activated_tracks.append(track)
            self._update_track_info(track.id, det)

        # Expire old gallery entries
        if self.reid_enabled:
            self.reid_gallery = [
                t for t in self.reid_gallery
                if self.frame_count - self.track_info.get(t.id, {}).get('last_seen', 0) <= self.reid_buffer
            ]

        # Update track lists
        refind_ids = {t.id for t in refind_tracks}
        removed_ids = {t.id for t in removed_tracks}
        self.tracked_tracks = activated_tracks + refind_tracks
        self.lost_tracks = [
            t for t in lost_tracks
            if t.id not in refind_ids and t.id not in removed_ids
        ]
        self.removed_tracks.extend(removed_tracks)

        # Build output
        output = []
        for track in self.tracked_tracks:
            bbox = track.get_state()[0]
            tid  = track.id
            output.append({
                'track_id':    tid,
                'bbox':        [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                'label':       self.track_labels.get(tid, 'unknown'),
                'confidence':  self.track_info.get(tid, {}).get('confidence', 0.0),
                'hits':        track.hits,
                'age':         track.age,
                'tracklet_len': track.tracklet_len
            })
        return output

    # ------------------------------------------------------------------

    def _update_track_info(self, track_id: int, det: Dict):
        """Update label and metadata for a track."""
        self.track_labels[track_id] = det['label']
        if track_id in self.track_info:
            self.track_info[track_id]['last_seen']  = self.frame_count
            self.track_info[track_id]['confidence'] = max(
                self.track_info[track_id]['confidence'],
                det['confidence']
            )
        else:
            self.track_info[track_id] = {
                'first_seen': self.frame_count,
                'last_seen':  self.frame_count,
                'label':      det['label'],
                'confidence': det['confidence']
            }

    def get_unique_items(self) -> Dict[int, Dict]:
        return self.track_info


# Alias for compatibility
ObjectTracker = ByteTracker
FeatureExtractor = DeepFeatureExtractor   # backwards-compat alias