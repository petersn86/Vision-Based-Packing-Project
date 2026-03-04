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
# STAGE 2 — Human confirmation via web UI:
#   Once an armed item has been absent from the box for
#   `absence_threshold` consecutive frames, a confirmation request is
#   posted to the shared `confirmation_queue` dict and a frame image
#   is saved. The Flask app surfaces this to the user as a notification
#   card with Yes/No buttons.
#
# FIXES:
#   - hand_flagged is no longer cleared while a hand is still present
#     in the frame, preventing scissors/items being grabbed from losing
#     their armed state mid-pickup.
#   - ever_hand_flagged persists across resets so the end-of-video flush
#     catches items that were touched even if absent_frames is 0 at EOV.
##############################################################################

from __future__ import annotations

import cv2
import numpy as np
import logging
import base64
import time
from typing  import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared confirmation queue  (imported by app.py and main.py)
# ─────────────────────────────────────────────────────────────────────────────

confirmation_queue: Dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitCandidate:
    """Tracks the exit-verification state of a single registered item."""
    instance_id:     int
    label:           str
    box_id:          str
    track_ids:       set              = field(default_factory=set)

    # Stage-1: was a hand overlapping this item?
    hand_flagged:      bool           = False
    hand_flag_frame:   Optional[int]  = None
    ever_hand_flagged: bool           = False  # survives resets — used by end-of-video flush

    # Stage-2: how many consecutive frames has this item been absent?
    absent_frames:   int              = 0

    # Verification gate: True once we've sent a confirmation request to the user
    user_queried:    bool             = False
    confirmation_id: Optional[str]   = None

    # Final verdict
    confirmed_removed: bool           = False
    removed_frame:     Optional[int]  = None
    removed_ts:        Optional[float]= None


# ─────────────────────────────────────────────────────────────────────────────
# Human verifier
# ─────────────────────────────────────────────────────────────────────────────

class HumanVerifier:
    """
    Saves a frame image and posts a confirmation request to confirmation_queue.
    The pipeline checks whether the user has answered before marking removal.
    """

    def __init__(self, crop_save_dir: str = "data/exit_crops"):
        import pathlib
        self.crop_dir = pathlib.Path(crop_save_dir)
        self.crop_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        """Always True — human is always available."""
        return True

    def request_confirmation(
        self,
        box_crop:     np.ndarray,
        item_label:   str,
        box_id:       str,
        instance_id:  int,
        frame_number: int,
        timestamp:    float,
    ) -> str:
        conf_id = f"exit_{instance_id}_{frame_number}"

        crop_path = self.crop_dir / f"{conf_id}.jpg"
        cv2.imwrite(str(crop_path), box_crop)

        _, buf = cv2.imencode(".jpg", box_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        confirmation_queue[conf_id] = {
            "instance_id": instance_id,
            "label":       item_label,
            "box_id":      box_id,
            "frame":       frame_number,
            "timestamp":   timestamp,
            "image_b64":   img_b64,
            "answer":      None,
            "asked_at":    time.time(),
        }

        logger.info(
            f"[EXIT-HUMAN] Confirmation requested for "
            f"'{item_label}' (#{instance_id}) frame {frame_number} id={conf_id}"
        )
        return conf_id

    def check_answer(self, confirmation_id: str) -> Optional[bool]:
        """Returns True (confirmed), False (rejected), or None (waiting)."""
        entry = confirmation_queue.get(confirmation_id)
        if entry is None:
            return None
        return entry["answer"]


# ─────────────────────────────────────────────────────────────────────────────
# Main ExitDetector
# ─────────────────────────────────────────────────────────────────────────────

class ExitDetector:
    """
    Two-stage exit detector. Stage 2 uses human confirmation via web UI.
    """

    def __init__(
        self,
        absence_threshold:      int   = 8,
        hand_overlap_threshold: float = 0.30,
        geometric_threshold:    int   = 15,
        crop_save_dir:          str   = "data/exit_crops",
        # Legacy LLaMA args kept for config.yaml compatibility — ignored
        llama_model:            str   = "",
        llama_enabled:          bool  = False,
        llama_max_retries:      int   = 2,
        geometric_fallback:     bool  = True,
    ):
        self.absence_threshold      = absence_threshold
        self.hand_overlap_threshold = hand_overlap_threshold
        self.geometric_threshold    = geometric_threshold
        self.geometric_fallback     = geometric_fallback

        self._verifier  = HumanVerifier(crop_save_dir)

        self._candidates: Dict[int, ExitCandidate] = {}
        self._box_bboxes: Dict[str, List[int]]     = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def sync_registry(self, registry) -> None:
        """
        Ensure every 'in_box' item in the registry has an ExitCandidate.
        Call once per frame after entry detection.
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
                self._candidates[inst.instance_id].track_ids = set(inst.track_ids)

    def update_hand_flags(
        self,
        active_track_bboxes: Dict[int, List[int]],
        hand_bboxes:         List[List[int]],
        frame_number:        int,
        hand_detected:       bool = False,
    ) -> List[int]:
        """Stage-1: arm candidates whose bbox overlaps a hand."""
        newly_flagged = []

        for cand in self._candidates.values():
            if cand.confirmed_removed or cand.hand_flagged:
                continue

            item_bbox  = None
            for tid in cand.track_ids:
                if tid in active_track_bboxes:
                    item_bbox = active_track_bboxes[tid]
                    break

            armed      = False
            arm_reason = ""

            # Mechanism 1: spatial overlap
            if item_bbox is not None:
                for hbbox in hand_bboxes:
                    if self._overlap_fraction(item_bbox, hbbox) >= self.hand_overlap_threshold:
                        armed      = True
                        arm_reason = "hand bbox overlaps item bbox"
                        break

            # Mechanism 2: hand visible but item not — likely being carried out
            if not armed and hand_detected:
                if item_bbox is not None:
                    armed      = True
                    arm_reason = "hand in frame, item visible"
                elif cand.absent_frames <= 5:
                    armed      = True
                    arm_reason = f"hand in frame, item absent {cand.absent_frames} frames"

            if armed:
                cand.hand_flagged      = True
                cand.hand_flag_frame   = frame_number
                cand.ever_hand_flagged = True   # permanent record — never cleared
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
        hand_detected:           bool = False,
    ) -> List[Tuple[int, str, str]]:
        """
        Stage-2: count absences, send human confirmation when threshold reached,
        return confirmed removals once user responds.

        Args:
            active_track_ids_in_box: Set of track_ids currently inside ANY box.
            box_bboxes:              {box_id: [x1,y1,x2,y2]} for visible boxes.
            frame:                   Full BGR frame.
            frame_number:            Current frame index.
            timestamp:               Frame timestamp in seconds.
            registry:                ItemRegistry (kept for API compatibility).
            hand_detected:           Whether a hand is present in this frame.
                                     When True, hand_flagged is NOT cleared on
                                     re-detection so items being actively grabbed
                                     stay armed.
        """
        self._box_bboxes.update(box_bboxes)
        confirmed_removals = []

        for cand in list(self._candidates.values()):
            if cand.confirmed_removed:
                continue

            item_in_box = bool(cand.track_ids & active_track_ids_in_box)

            if item_in_box:
                if not cand.user_queried:
                    cand.absent_frames = 0
                    # Only un-arm if no hand is currently present.
                    # If a hand is in the frame the item may be mid-grab —
                    # keep it armed so the absence counter fires after pickup.
                    if not hand_detected:
                        cand.hand_flagged = False
                continue

            cand.absent_frames += 1

            effective_threshold = (
                self.absence_threshold if cand.hand_flagged
                else self.geometric_threshold
            )

            logger.debug(
                f"[EXIT] #{cand.instance_id} '{cand.label}' absent "
                f"{cand.absent_frames}/{effective_threshold} frames "
                f"(hand_flagged={cand.hand_flagged})"
            )

            # ── Already waiting for user answer ──────────────────────────
            if cand.user_queried and cand.confirmation_id:
                answer = self._verifier.check_answer(cand.confirmation_id)

                if answer is True:
                    cand.confirmed_removed = True
                    cand.removed_frame     = frame_number
                    cand.removed_ts        = timestamp
                    confirmed_removals.append(
                        (cand.instance_id, cand.label, cand.box_id)
                    )
                    logger.info(
                        f"[EXIT] ✓ USER confirmed removal: "
                        f"#{cand.instance_id} '{cand.label}' from {cand.box_id} "
                        f"at t={timestamp:.2f}s"
                    )

                elif answer is False:
                    logger.info(
                        f"[EXIT] ✗ USER rejected exit: "
                        f"#{cand.instance_id} '{cand.label}' still present"
                    )
                    cand.absent_frames   = 0
                    cand.user_queried    = False
                    cand.confirmation_id = None
                    cand.hand_flagged    = False
                    # Note: ever_hand_flagged is NOT cleared here

                continue

            # ── Threshold reached — ask the user ─────────────────────────
            if cand.absent_frames >= effective_threshold and not cand.user_queried:
                conf_id = self._verifier.request_confirmation(
                    box_crop     = frame,
                    item_label   = cand.label,
                    box_id       = cand.box_id,
                    instance_id  = cand.instance_id,
                    frame_number = frame_number,
                    timestamp    = timestamp,
                )
                cand.user_queried    = True
                cand.confirmation_id = conf_id

        return confirmed_removals

    def get_candidates(self) -> Dict[int, ExitCandidate]:
        return self._candidates

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _overlap_fraction(bbox_a: List[int], bbox_b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter  = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        return inter / area_a

    @staticmethod
    def _crop_frame(frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        h, w   = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else frame