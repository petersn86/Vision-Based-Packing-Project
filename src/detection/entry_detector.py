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
# UPDATED: get_active_track_ids_in_box() now accepts current_frame
#          and evicts tracks that haven't been seen by YOLO in
#          stale_threshold frames. This allows the ExitDetector to
#          correctly start counting absences for items that simply
#          vanish from YOLO detection (e.g. removed early in video)
#          rather than staying "confirmed inside" forever.
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
                 entry_threshold: int  = 3,
                 exit_threshold:  int  = 5,
                 require_motion:  bool = False):
        """
        Args:
            entry_threshold: Consecutive frames inside box to confirm entry.
            exit_threshold:  Frames outside box before the EntryDetector considers
                             the (track_id, box_id) pair unconfirmed again.
                             NOTE: actual item *removal* is handled by ExitDetector.
            require_motion:  If True, only detect items that physically move INTO boxes.
                             If False, also detect items already present in boxes.
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold  = exit_threshold
        self.require_motion  = require_motion

        # track_id -> list of {frame, bbox, box_id, overlap, inside}
        self.track_history = defaultdict(list)

        # track_id -> box_id  (confirmed entries)
        self.entered_items = {}

        # (track_id, box_id) pairs that are confirmed inside
        self.confirmed_entries: Set[Tuple[int, str]] = set()

        # consecutive-frame counters
        self.inside_counter:  Dict[Tuple[int, str], int] = defaultdict(int)
        self.outside_counter: Dict[Tuple[int, str], int] = defaultdict(int)

        # track_id -> last frame on which detect_entry() was called for it
        # Used by get_active_track_ids_in_box() to evict stale tracks
        self._last_seen_frame: Dict[int, int] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_item_center(self, bbox: List[int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def is_point_in_box(self, point: Tuple[int, int], box_bbox: List[int]) -> bool:
        x, y = point
        x1, y1, x2, y2 = box_bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def get_overlap_percentage(self, item_bbox: List[int], box_bbox: List[int]) -> float:
        """
        Fraction of the item bbox that overlaps the box bbox.
        Returns [0.0, 1.0].
        """
        ix1, iy1, ix2, iy2 = item_bbox
        bx1, by1, bx2, by2 = box_bbox

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

    # ──────────────────────────────────────────────────────────────────────
    # Entry detection
    # ──────────────────────────────────────────────────────────────────────

    def detect_entry(self,
                     track_id:          int,
                     item_bbox:         List[int],
                     box_id:            str,
                     box_bbox:          List[int],
                     frame_number:      int,
                     overlap_threshold: float = 0.5) -> Optional[str]:
        """
        Detect if an item is inside a box.

        Returns:
            box_id if entry is *newly* confirmed this frame, else None.
        """
        key = (track_id, box_id)

        # Record that this track was seen this frame
        self._last_seen_frame[track_id] = frame_number

        overlap   = self.get_overlap_percentage(item_bbox, box_bbox)
        is_inside = overlap >= overlap_threshold

        # Update history
        self.track_history[track_id].append({
            'frame':   frame_number,
            'bbox':    item_bbox,
            'box_id':  box_id,
            'overlap': overlap,
            'inside':  is_inside,
        })
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id] = self.track_history[track_id][-30:]

        # Already confirmed — maintain outside counter for internal housekeeping
        # (ExitDetector drives the actual removal decision)
        if key in self.confirmed_entries:
            if not is_inside:
                self.outside_counter[key] += 1
                if self.outside_counter[key] >= self.exit_threshold:
                    # Unconfirm so it can re-confirm if item re-enters
                    self.confirmed_entries.discard(key)
                    self.inside_counter[key]  = 0
                    self.outside_counter[key] = 0
            else:
                self.outside_counter[key] = 0
            return None

        # Track consecutive frames inside box
        if is_inside:
            self.inside_counter[key]  += 1
            self.outside_counter[key]  = 0

            if self.inside_counter[key] >= self.entry_threshold:
                if self.require_motion:
                    if self._verify_entry_motion(track_id, box_id, box_bbox):
                        self.confirmed_entries.add(key)
                        self.entered_items[track_id] = box_id
                        return box_id
                else:
                    self.confirmed_entries.add(key)
                    self.entered_items[track_id] = box_id
                    return box_id
        else:
            self.inside_counter[key]  = 0
            self.outside_counter[key] += 1

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Helpers for ExitDetector integration
    # ──────────────────────────────────────────────────────────────────────

    def notify_track_seen(self, track_id: int, frame_number: int) -> None:
        """
        Notify the EntryDetector that a track was seen by YOLO this frame,
        even if it wasn't inside a box. This keeps _last_seen_frame current
        so stale eviction doesn't fire prematurely on items that are still
        visible but momentarily outside the box region (e.g. being picked up).
        """
        self._last_seen_frame[track_id] = frame_number

    def get_active_track_ids_in_box(self,
                                     current_frame:   int = None,
                                     stale_threshold: int = 5) -> Set[int]:
        """
        Return the set of track_ids currently confirmed inside any box.

        This is the primary signal ExitDetector uses to count absences:
        if a registered item's track_id is NOT in this set, it may have been removed.

        Args:
            current_frame:    Current frame index. When provided, any confirmed track
                              that hasn't been seen by detect_entry() for stale_threshold
                              or more frames is evicted from confirmed_entries. This
                              ensures that items which simply vanish from YOLO detection
                              (rather than being explicitly detected outside the box)
                              are treated as absent by the ExitDetector.
            stale_threshold:  Number of frames without a detect_entry() call before a
                              track is considered stale. At 0.5s frame interval, a value
                              of 5 = ~2.5 seconds of no detection = start counting absence.
        """
        if current_frame is not None:
            stale_keys = [
                (tid, bid)
                for (tid, bid) in self.confirmed_entries
                if (current_frame - self._last_seen_frame.get(tid, current_frame)) >= stale_threshold
            ]
            for key in stale_keys:
                self.confirmed_entries.discard(key)
                self.inside_counter[key]  = 0
                self.outside_counter[key] = 0

        return {tid for (tid, _) in self.confirmed_entries}

    def get_active_track_ids_for_box(self, box_id: str) -> Set[int]:
        """Return confirmed track_ids for a specific box."""
        return {tid for (tid, bid) in self.confirmed_entries if bid == box_id}

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _verify_entry_motion(self,
                              track_id: int,
                              box_id:   str,
                              box_bbox: List[int]) -> bool:
        """
        Verify item moved INTO the box (used only when require_motion=True).
        """
        history = self.track_history[track_id]

        if len(history) < self.entry_threshold + 2:
            return False

        recent = history[-(self.entry_threshold + 5):]
        half   = len(recent) // 2

        was_outside = any(
            r['box_id'] == box_id and not r['inside']
            for r in recent[:half]
        )
        now_inside = any(
            r['box_id'] == box_id and r['inside']
            for r in recent[half:]
        )
        return was_outside and now_inside

    # ──────────────────────────────────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────────────────────────────────

    def get_box_for_item(self, track_id: int) -> Optional[str]:
        return self.entered_items.get(track_id)

    def get_all_entries(self) -> Dict[str, List[int]]:
        entries_by_box: Dict[str, List[int]] = defaultdict(list)
        for (tid, bid) in self.confirmed_entries:
            entries_by_box[bid].append(tid)
        return dict(entries_by_box)

    def reset(self):
        self.track_history.clear()
        self.entered_items.clear()
        self.confirmed_entries.clear()
        self.inside_counter.clear()
        self.outside_counter.clear()
        self._last_seen_frame.clear()