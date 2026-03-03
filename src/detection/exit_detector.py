##############################################################################
# @Author: Peter Nolan
# @Document: 'exit_detector.py'
#
# Description:
# Two-stage exit detection for the Vision-Based Packing Project.
#
# STAGE 1 — Hand-overlap trigger (cheap, runs every frame):
#   When a confirmed in-box item's bounding box significantly overlaps
#   with a detected hand bounding box, the item is flagged as a
#   "pickup candidate".  The candidate is NOT immediately removed —
#   it just arms the verifier.
#
# STAGE 2 — LLaMA visual verification (expensive, runs only when needed):
#   Once an armed item has been *absent* from the box for
#   `absence_threshold` consecutive frames, LLaMA is asked to look at
#   the full box-region crop and decide whether the item is still there.
#   Only if LLaMA confirms absence does the item get marked "removed".
#
# Fallback (no hand model / no LLaMA):
#   Pure geometric exit — item is removed after `absence_threshold`
#   consecutive frames outside the box, with no secondary verification.
#
# The ExitDetector is designed to slot alongside the existing EntryDetector
# and ItemRegistry without requiring changes to the core tracking loop.
##############################################################################

from __future__ import annotations

import cv2
import numpy as np
import logging
import ollama
from typing  import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitCandidate:
    """Tracks the exit-verification state of a single registered item."""
    instance_id:    int
    label:          str
    box_id:         str
    track_ids:      set               = field(default_factory=set)

    # Stage-1: was a hand overlapping this item?
    hand_flagged:   bool              = False
    hand_flag_frame: Optional[int]   = None

    # Stage-2: how many consecutive frames has this item been absent?
    absent_frames:  int               = 0

    # Verification gate: has LLaMA been asked yet this absence run?
    llama_queried:  bool              = False

    # Final verdict
    confirmed_removed: bool           = False
    removed_frame:     Optional[int]  = None
    removed_ts:        Optional[float]= None


# ─────────────────────────────────────────────────────────────────────────────
# LLaMA verifier (stateless helper)
# ─────────────────────────────────────────────────────────────────────────────

class ExitVerifier:
    """
    Asks LLaMA whether a specific item is still visible inside a box crop.

    The crop is the bounding-box region of the *box* (not just the item),
    giving LLaMA full context of the box contents.
    """

    def __init__(self, model_name: str = "llama3.2-vision", max_retries: int = 2):
        self.model_name  = model_name
        self.max_retries = max_retries
        self._available  = False

        try:
            ollama.show(self.model_name)
            self._available = True
            logger.info(f"[EXIT] LLaMA verifier ready: {model_name}")
        except Exception as e:
            logger.warning(f"[EXIT] LLaMA not available for exit verification: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def _build_prompt(self, item_label: str, box_items: List[str]) -> str:
        other = [i for i in box_items if i.lower() != item_label.lower()]
        others_str = ", ".join(other) if other else "none"
        return (
            "You are a packing inventory verifier checking whether an item is still inside a cardboard box.\n\n"
            "Look carefully at the image. It shows the interior of a cardboard box viewed from above.\n\n"
            f"QUESTION: Is a '{item_label}' physically present inside this box?\n\n"
            f"Other items that may still be in the box: {others_str}\n\n"
            "RULES:\n"
            "1. Reply with ONLY one word: YES or NO\n"
            "2. YES = you can clearly see a '{item_label}' sitting inside the box\n"
            "3. NO  = the box appears empty, OR the item is not visible, OR you are not sure\n"
            "4. An empty cardboard interior with no objects = NO\n"
            "5. Do not explain. Do not hedge. Just YES or NO."
        )

    def verify_item_present(
        self,
        box_crop:   np.ndarray,
        item_label: str,
        box_items:  List[str],
    ) -> bool:
        """
        Returns True if LLaMA believes the item is still in the box.
        Returns True (conservative / no-removal) if LLaMA is unavailable or errors.
        """
        if not self._available:
            return True   # conservative: don't remove without verification

        _, buf = cv2.imencode(".jpg", box_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_bytes = buf.tobytes()

        prompt = self._build_prompt(item_label, box_items)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{
                        "role":    "user",
                        "content": prompt,
                        "images":  [img_bytes],
                    }],
                )
                raw = response["message"]["content"].strip().upper()
                # Accept any response that starts with YES / NO
                if raw.startswith("YES"):
                    logger.info(f"[EXIT-LLAMA] '{item_label}' → PRESENT (attempt {attempt})")
                    return True
                if raw.startswith("NO"):
                    logger.info(f"[EXIT-LLAMA] '{item_label}' → ABSENT (attempt {attempt})")
                    return False

                logger.warning(f"[EXIT-LLAMA] Unexpected response: '{raw}' (attempt {attempt})")

            except Exception as e:
                logger.warning(f"[EXIT-LLAMA] Error on attempt {attempt}: {e}")

        # All retries failed — be conservative, assume still present
        logger.warning(f"[EXIT-LLAMA] All retries failed for '{item_label}', assuming PRESENT")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Main ExitDetector
# ─────────────────────────────────────────────────────────────────────────────

class ExitDetector:
    """
    Two-stage exit detector.

    Usage (inside main frame loop):
    ──────────────────────────────
        exit_detector = ExitDetector(...)

        for frame_data in frames:
            frame      = frame_data['frame']
            frame_idx  = frame_data['index']
            timestamp  = frame_data['timestamp']

            # 1. After entry detection / registry updates:
            exit_detector.sync_registry(registry)

            # 2. Stage-1: hand-overlap check (pass None if hand bbox unavailable)
            exit_detector.update_hand_flags(
                active_track_bboxes,   # {track_id: [x1,y1,x2,y2]}
                hand_bboxes,           # list of [x1,y1,x2,y2]  (may be [])
                frame_idx,
            )

            # 3. Stage-2: absence tracking + LLaMA verification
            removals = exit_detector.update_absences(
                active_track_ids_in_box,   # set of track_ids currently in any box
                box_bboxes,                # {box_id: [x1,y1,x2,y2]}
                frame,
                frame_idx,
                timestamp,
                registry,
            )

            # 4. Apply confirmed removals to the registry
            for instance_id, label, box_id in removals:
                registry.mark_removed(instance_id, frame_idx, timestamp)
                logger.info(f"[EXIT] Confirmed: '{label}' removed from {box_id}")
    """

    def __init__(
        self,
        absence_threshold:       int   = 8,
        hand_overlap_threshold:  float = 0.30,
        llama_model:             str   = "llama3.2-vision",
        llama_enabled:           bool  = True,
        llama_max_retries:       int   = 2,
        geometric_fallback:      bool  = True,
        geometric_threshold:     int   = 15,
    ):
        """
        Args:
            absence_threshold:      Frames a hand-flagged item must be absent before
                                    LLaMA verification fires. Acts as the *fast* path
                                    for items a hand was seen touching.
            hand_overlap_threshold: Fraction of item bbox that must overlap a hand bbox
                                    to arm the fast path.
            llama_model:            Ollama model name for verification.
            llama_enabled:          If False, skip LLaMA (geometric threshold only).
            llama_max_retries:      Retry count for failed LLaMA calls.
            geometric_fallback:     Kept for config compatibility; always True internally.
            geometric_threshold:    Frames ANY item (hand-flagged or not) must be absent
                                    before LLaMA verification fires. Acts as the *slow*
                                    path for items never seen being touched. Should be
                                    larger than absence_threshold.
        """
        self.absence_threshold      = absence_threshold
        self.hand_overlap_threshold = hand_overlap_threshold
        self.geometric_fallback     = geometric_fallback
        self.geometric_threshold    = geometric_threshold

        self._verifier = ExitVerifier(llama_model, llama_max_retries) if llama_enabled else None

        # instance_id → ExitCandidate
        self._candidates: Dict[int, ExitCandidate] = {}

        # box_id → last known bbox
        self._box_bboxes: Dict[str, List[int]] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def sync_registry(self, registry) -> None:
        """
        Ensure every 'in_box' item in the registry has an ExitCandidate entry.
        Also keeps track_ids current as the tracker assigns new IDs over time.
        Call this once per frame *after* entry detection.
        """
        for inst in registry.get_active_items():
            if inst.instance_id not in self._candidates:
                self._candidates[inst.instance_id] = ExitCandidate(
                    instance_id = inst.instance_id,
                    label       = inst.refined_label or inst.label,
                    box_id      = inst.box_id,
                    track_ids   = set(inst.track_ids),
                )
                logger.debug(f"[EXIT] Tracking candidate: #{inst.instance_id} '{inst.refined_label}'")
            else:
                # Keep track_ids current — tracker may assign new IDs over time
                self._candidates[inst.instance_id].track_ids = set(inst.track_ids)

    def update_hand_flags(
        self,
        active_track_bboxes: Dict[int, List[int]],
        hand_bboxes:         List[List[int]],
        frame_number:        int,
        hand_detected:       bool = False,
    ) -> List[int]:
        """
        Stage-1: check whether any registered item's bbox overlaps a hand.

        Two arming mechanisms:
        1. Spatial overlap: item bbox overlaps a hand bbox by >= hand_overlap_threshold
        2. Frame-level fallback: if hand_detected=True but no item bbox is visible
           (item being actively lifted out of box), arm all candidates whose tracks
           were last seen this frame or very recently.

        Args:
            active_track_bboxes: {track_id: [x1,y1,x2,y2]} for all currently
                                  visible tracked objects.
            hand_bboxes:         List of hand bounding boxes this frame.
            frame_number:        Current frame index.
            hand_detected:       Bool from HandDetector.detect() — True if any hand
                                 present in the frame, regardless of spatial overlap.

        Returns:
            List of instance_ids that were newly hand-flagged this frame.
        """
        newly_flagged = []

        # Nothing to do if no hand signal at all
        if not hand_bboxes and not hand_detected:
            return newly_flagged

        for cand in self._candidates.values():
            if cand.confirmed_removed or cand.hand_flagged:
                continue

            # Find the most recent track_id for this candidate that's visible
            item_bbox = None
            for tid in cand.track_ids:
                if tid in active_track_bboxes:
                    item_bbox = active_track_bboxes[tid]
                    break

            armed = False
            arm_reason = ""

            if item_bbox is not None and hand_bboxes:
                # Primary path: spatial overlap
                for hand_bbox in hand_bboxes:
                    overlap = self._bbox_overlap_fraction(item_bbox, hand_bbox)
                    if overlap >= self.hand_overlap_threshold:
                        armed = True
                        arm_reason = f"spatial overlap {overlap:.2f}"
                        break

            if not armed and hand_detected:
                # Fallback: hand is in frame and item is either visible or recently absent.
                # We trust hand_detected=True as a frame-level signal — any in-box item
                # in a frame where a hand is present is a candidate for removal.
                # LLaMA verification prevents false positives from this broad arming.
                if item_bbox is not None:
                    # Item visible same frame as hand — may be mid-grab
                    armed = True
                    arm_reason = "hand in frame, item visible"
                elif cand.absent_frames <= 5:
                    # Item just disappeared while hand was present
                    armed = True
                    arm_reason = f"hand in frame, item absent {cand.absent_frames} frames"

            if armed:
                cand.hand_flagged    = True
                cand.hand_flag_frame = frame_number
                newly_flagged.append(cand.instance_id)
                logger.info(
                    f"[EXIT-STAGE1] Armed #{cand.instance_id} '{cand.label}' "
                    f"({arm_reason}, frame {frame_number})"
                )

        return newly_flagged

    def update_absences(
        self,
        active_track_ids_in_box: set,
        box_bboxes:              Dict[str, List[int]],
        frame:                   np.ndarray,
        frame_number:            int,
        timestamp:               float,
        registry,
    ) -> List[Tuple[int, str, str]]:
        """
        Stage-2: count absences, trigger LLaMA when threshold reached.

        Args:
            active_track_ids_in_box: Set of track_ids currently inside ANY box this frame.
            box_bboxes:              {box_id: [x1,y1,x2,y2]} for current visible boxes.
            frame:                   Full BGR frame (for cropping box region).
            frame_number:            Current frame index.
            timestamp:               Frame timestamp in seconds.
            registry:                ItemRegistry — used to get current box contents.

        Returns:
            List of (instance_id, label, box_id) tuples for confirmed removals.
            Caller is responsible for calling registry.mark_removed() on each.
        """
        self._box_bboxes.update(box_bboxes)
        confirmed_removals = []

        for cand in list(self._candidates.values()):
            if cand.confirmed_removed:
                continue

            # Check if any of this candidate's track_ids are active in a box
            still_visible = bool(cand.track_ids & active_track_ids_in_box)

            if still_visible:
                # Item is back — reset absence counter and LLaMA gate
                cand.absent_frames  = 0
                cand.llama_queried  = False
                continue

            cand.absent_frames += 1

            # Effective threshold:
            #   - Hand-flagged items: use absence_threshold (fast path — high confidence signal)
            #   - Non-hand-flagged:   use geometric_threshold (slow path — conservative)
            #     Non-flagged items need a LOT of absent frames because YOLO frequently
            #     drops stationary items for many frames even when they're still present.
            if cand.hand_flagged:
                effective_threshold = self.absence_threshold
            else:
                effective_threshold = self.geometric_threshold

            logger.debug(
                f"[EXIT] #{cand.instance_id} '{cand.label}' absent "
                f"{cand.absent_frames}/{effective_threshold} frames "
                f"(hand_flagged={cand.hand_flagged})"
            )

            # Trigger LLaMA verification once per absence run
            if cand.absent_frames >= effective_threshold and not cand.llama_queried:
                cand.llama_queried = True

                box_bbox   = self._box_bboxes.get(cand.box_id)
                box_items  = registry.get_unique_labels_for_box(cand.box_id)
                item_present = True  # default: conservative

                if box_bbox is not None and self._verifier is not None and self._verifier.available:
                    box_crop   = self._crop_frame(frame, box_bbox)
                    item_present = self._verifier.verify_item_present(
                        box_crop, cand.label, box_items
                    )
                    logger.info(
                        f"[EXIT-STAGE2] LLaMA says '{cand.label}' "
                        f"{'PRESENT' if item_present else 'ABSENT'} in {cand.box_id} "
                        f"(frame {frame_number})"
                    )
                elif box_bbox is None:
                    logger.warning(
                        f"[EXIT] No bbox for {cand.box_id}, "
                        f"cannot crop for LLaMA verification of '{cand.label}'"
                    )
                    # No box visible — assume item truly gone
                    item_present = False

                if not item_present:
                    cand.confirmed_removed = True
                    cand.removed_frame     = frame_number
                    cand.removed_ts        = timestamp
                    confirmed_removals.append((cand.instance_id, cand.label, cand.box_id))
                    logger.info(
                        f"[EXIT] ✓ CONFIRMED removal: #{cand.instance_id} "
                        f"'{cand.label}' from {cand.box_id} at t={timestamp:.2f}s"
                    )
                else:
                    # LLaMA says item is still there — reset counters
                    # (detection gap / occlusion, not a real removal)
                    logger.info(
                        f"[EXIT] ✗ FALSE EXIT rejected: #{cand.instance_id} "
                        f"'{cand.label}' still present per LLaMA"
                    )
                    cand.absent_frames  = 0
                    cand.llama_queried  = False
                    cand.hand_flagged   = False  # un-arm; needs new hand touch to re-arm



        return confirmed_removals

    def get_candidates(self) -> Dict[int, ExitCandidate]:
        """Return all exit candidates (for debugging / annotation)."""
        return self._candidates

    def get_armed_candidates(self) -> List[ExitCandidate]:
        """Return candidates that are hand-flagged but not yet confirmed removed."""
        return [c for c in self._candidates.values()
                if c.hand_flagged and not c.confirmed_removed]

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _bbox_overlap_fraction(item_bbox: List[int], hand_bbox: List[int]) -> float:
        """
        Fraction of the *item* bbox that is covered by the hand bbox.
        Returns [0.0, 1.0].
        """
        ix1, iy1, ix2, iy2 = item_bbox
        hx1, hy1, hx2, hy2 = hand_bbox

        ox1 = max(ix1, hx1)
        oy1 = max(iy1, hy1)
        ox2 = min(ix2, hx2)
        oy2 = min(iy2, hy2)

        if ox2 <= ox1 or oy2 <= oy1:
            return 0.0

        intersection = (ox2 - ox1) * (oy2 - oy1)
        item_area    = max(1, (ix2 - ix1) * (iy2 - iy1))
        return intersection / item_area

    @staticmethod
    def _crop_frame(frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Crop to the box interior.

        We shrink inward slightly (instead of padding outward) so the crop
        shows only the inside of the box — not surrounding clutter like walls,
        floors, or other objects beside the box. This gives LLaMA a cleaner,
        less confusing view when checking if an item is still present.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        # Inset by 8% of each dimension to clip the box walls/flaps
        inset_x = max(5, int(box_w * 0.08))
        inset_y = max(5, int(box_h * 0.08))
        x1 = min(w - 1, x1 + inset_x)
        y1 = min(h - 1, y1 + inset_y)
        x2 = max(0,     x2 - inset_x)
        y2 = max(0,     y2 - inset_y)
        # Guard against degenerate bbox
        if x2 <= x1 or y2 <= y1:
            return frame[max(0,bbox[1]):min(h,bbox[3]), max(0,bbox[0]):min(w,bbox[2])]
        return frame[y1:y2, x1:x2]