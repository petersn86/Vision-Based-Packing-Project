##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'box_tracker.py'
#
# Description:
# Assigns stable IDs to YOLO-detected boxes across video frames.
#
# When a barcode/QR code is visible on a box, the scanner's value
# becomes that box's ID. When the code is hidden or blurry the
# BoxTracker propagates the last-known ID using IoU matching,
# so the rest of the pipeline always gets a meaningful box identity
# rather than the old 'BOX-001' placeholder.
#
# ID priority (highest → lowest):
#   1. Live barcode scan result for this frame
#   2. Last confirmed barcode for this tracked box
#   3. Auto-generated fallback ID:  "BOX-<n>"
##############################################

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: List[int], b: List[int]) -> float:
    """Compute Intersection over Union for two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal track record
# ─────────────────────────────────────────────────────────────────────────────

class _BoxTrack:
    _next_id = 1

    def __init__(self, bbox: List[int], barcode_id: Optional[str], frame_idx: int):
        self.internal_id:  int          = _BoxTrack._next_id
        _BoxTrack._next_id += 1

        self.bbox:          List[int]        = bbox
        self.barcode_id:    Optional[str]    = barcode_id   # None until scanned
        self.fallback_id:   str              = f"BOX-{self.internal_id:03d}"
        self.last_frame:    int              = frame_idx
        self.first_frame:   int              = frame_idx
        self.age:           int              = 0            # frames tracked
        self.misses:        int              = 0            # consecutive un-matched frames

    @property
    def box_id(self) -> str:
        """The ID that the pipeline should use for this box."""
        return self.barcode_id if self.barcode_id else self.fallback_id

    def update(self, bbox: List[int], barcode_id: Optional[str], frame_idx: int):
        self.bbox       = bbox
        self.last_frame = frame_idx
        self.misses     = 0
        self.age       += 1
        if barcode_id and not self.barcode_id:
            # First time we see a code for this track – lock it in
            self.barcode_id = barcode_id
            logger.info(
                f"[BOX-TRACKER] Track #{self.internal_id} → "
                f"barcode resolved: '{barcode_id}'"
            )
        elif barcode_id and barcode_id != self.barcode_id:
            # Code changed (e.g. different box moved into frame)
            logger.warning(
                f"[BOX-TRACKER] Track #{self.internal_id} barcode changed: "
                f"'{self.barcode_id}' → '{barcode_id}' (keeping original)"
            )
            # Keep the original to avoid ID flicker; log for investigation


# ─────────────────────────────────────────────────────────────────────────────
# BoxTracker
# ─────────────────────────────────────────────────────────────────────────────

class BoxTracker:
    """
    Propagates stable box IDs across frames using IoU matching.

    Usage inside the main frame loop
    ──────────────────────────────────
        box_tracker = BoxTracker(iou_threshold=0.3, max_age=30)

        # After YOLO box detection + barcode scan:
        box_detections = box_tracker.update(
            raw_box_detections = box_detections_raw,   # from YOLO
            scanner            = barcode_scanner,
            frame_idx          = frame_idx,
        )

        # box_detections is the same list but each dict now has
        # a 'box_id' key with a real (or fallback) ID.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age:       int   = 30,
    ):
        self.iou_threshold = iou_threshold
        self.max_age       = max_age

        self._tracks: List[_BoxTrack] = []
        logger.info(
            f"BoxTracker initialised "
            f"(iou_threshold={iou_threshold}, max_age={max_age})"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Main update
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        raw_box_detections: List[Dict],
        scanner,            # BarcodeScanner instance (or None)
        frame_idx: int,
    ) -> List[Dict]:
        """
        Match YOLO box detections to existing tracks using IoU.
        Assign / propagate barcode IDs.

        Parameters
        ──────────
        raw_box_detections : list of dicts with at minimum a 'bbox' key
        scanner            : BarcodeScanner used to resolve box_id
        frame_idx          : current frame number

        Returns
        ───────
        Same list with a 'box_id' key added to every dict.
        """
        detections = raw_box_detections  # shorthand

        if not detections:
            # Increment miss counters, prune dead tracks
            for t in self._tracks:
                t.misses += 1
            self._tracks = [t for t in self._tracks if t.misses <= self.max_age]
            return []

        det_bboxes = [d["bbox"] for d in detections]

        # ── Step 1: build IoU matrix (tracks × detections) ──────────────
        if self._tracks:
            iou_matrix = np.zeros((len(self._tracks), len(detections)), dtype=np.float32)
            for ti, track in enumerate(self._tracks):
                for di, det_bbox in enumerate(det_bboxes):
                    iou_matrix[ti, di] = _iou(track.bbox, det_bbox)

            # Greedy match: highest IoU first
            matched_tracks = set()
            matched_dets   = set()
            # Sort all (iou, ti, di) descending
            pairs = sorted(
                [(iou_matrix[ti, di], ti, di)
                 for ti in range(len(self._tracks))
                 for di in range(len(detections))],
                key=lambda x: -x[0],
            )
            track_to_det: Dict[int, int] = {}
            det_to_track: Dict[int, int] = {}
            for iou_val, ti, di in pairs:
                if iou_val < self.iou_threshold:
                    break
                if ti in matched_tracks or di in matched_dets:
                    continue
                track_to_det[ti] = di
                det_to_track[di] = ti
                matched_tracks.add(ti)
                matched_dets.add(di)
        else:
            track_to_det = {}
            det_to_track = {}
            matched_tracks = set()

        # ── Step 2: update matched tracks ────────────────────────────────
        for ti, di in track_to_det.items():
            det       = detections[di]
            bbox      = det["bbox"]
            barcode   = (
                scanner.resolve_box_id(bbox, frame_idx)
                if scanner and scanner.enabled else None
            )
            self._tracks[ti].update(bbox, barcode, frame_idx)

        # ── Step 3: create new tracks for unmatched detections ───────────
        for di, det in enumerate(detections):
            if di in det_to_track:
                continue
            bbox    = det["bbox"]
            barcode = (
                scanner.resolve_box_id(bbox, frame_idx)
                if scanner and scanner.enabled else None
            )
            new_track = _BoxTrack(bbox, barcode, frame_idx)
            self._tracks.append(new_track)
            logger.debug(
                f"[BOX-TRACKER] New track #{new_track.internal_id} "
                f"id='{new_track.box_id}' frame={frame_idx}"
            )

        # ── Step 4: age out unmatched existing tracks ────────────────────
        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.max_age]

        # ── Step 5: build output ─────────────────────────────────────────
        # Rebuild det→track mapping after possible new-track insertion
        # We rely on the order: first len(old_tracks) entries are old tracks,
        # new tracks appended at the end. Reconstruct cleanly.
        final_tracks: Dict[int, _BoxTrack] = {}   # di → track
        # matched old tracks
        for ti, di in track_to_det.items():
            final_tracks[di] = self._tracks[ti]
        # new tracks (they were appended, find by internal_id)
        new_track_map = {t.internal_id: t for t in self._tracks}
        # Walk through unmatched dets in order to pair with newly created tracks
        unmatched_det_indices = [
            di for di in range(len(detections))
            if di not in det_to_track
        ]
        # New tracks appended in same order as unmatched_det_indices
        new_track_candidates = [
            t for t in self._tracks
            if t.first_frame == frame_idx and t.age == 0
        ]
        for di, track in zip(unmatched_det_indices, new_track_candidates):
            final_tracks[di] = track

        output = []
        for di, det in enumerate(detections):
            track  = final_tracks.get(di)
            box_id = track.box_id if track else f"BOX-UNK-{di}"
            output.append({**det, "box_id": box_id})

        return output

    # ──────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────

    def get_active_tracks(self) -> List[Dict]:
        return [
            {
                "internal_id": t.internal_id,
                "box_id":      t.box_id,
                "barcode_id":  t.barcode_id,
                "bbox":        t.bbox,
                "age":         t.age,
                "misses":      t.misses,
            }
            for t in self._tracks
        ]

    def reset(self):
        self._tracks.clear()
        _BoxTrack._next_id = 1
        logger.debug("[BOX-TRACKER] Reset.")