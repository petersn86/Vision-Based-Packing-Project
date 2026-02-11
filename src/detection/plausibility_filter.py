##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'plausibility_filter.py'
#
# Description:
# Filters YOLO detections that are implausible given their label
# and bounding box size relative to the frame.
#
# Large appliances (refrigerators, microwaves, ovens, etc.) physically
# cannot fit inside a packing box, so if YOLO detects one but its
# bounding box is small relative to the full frame, it is almost
# certainly a misclassification (e.g. a picture frame being mistaken
# for a fridge).
#
# Each "implausible label" has a minimum frame-fraction threshold.
# A detection is kept only if its bbox area >= (frame_area * threshold).
# If the bbox is too small for that label, it is discarded entirely —
# neither tracked nor passed to LLaMA.
#
##############################################

from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default minimum frame-fraction thresholds per label group.
#
# Format:  label (lowercase)  ->  min fraction of total frame area (0.0–1.0)
#
# Interpretation: a detection with this label is only accepted if its
# bounding box covers AT LEAST this fraction of the full frame.
#
# Rationale:
#   - A real refrigerator filmed from above would dominate the frame (≥60%)
#   - A real microwave or oven from above would still be large (≥40%)
#   - A TV/monitor is large but could be partially visible (≥25%)
#   - A washing machine / dryer would fill most of the frame (≥50%)
#
# These are intentionally conservative — err on the side of discarding
# false positives rather than keeping misclassified appliances.
# ---------------------------------------------------------------------------
DEFAULT_MIN_FRAME_FRACTION: Dict[str, float] = {
    # Kitchen appliances
    "refrigerator":     0.60,
    "fridge":           0.60,
    "microwave":        0.40,
    "oven":             0.50,
    "stove":            0.50,
    "toaster oven":     0.20,
    "dishwasher":       0.50,
    "range":            0.50,
    "sink":             0.35,

    # Laundry
    "washing machine":  0.50,
    "washer":           0.50,

    # Large electronics
    "television":       0.30,
    "tv":               0.30,
    "monitor":          0.25,
    "screen":           0.25,
    "projector":        0.25,

    # Furniture
    "couch":            0.40,
    "sofa":             0.40,
    "bed":              0.50,
    "desk":             0.35,
    "dining table":     0.40,
    "table":            0.35,
    "chair":            0.20,
    "wardrobe":         0.50,
    "cabinet":          0.35,

    # Vehicles / misc large objects
    "car":              0.50,
    "truck":            0.60,
    "bicycle":          0.30,
}


class PlausibilityFilter:
    """
    Filters YOLO detections whose bounding box is implausibly small
    for the detected label, relative to the full frame size.

    Usage:
        pf = PlausibilityFilter(thresholds=..., enabled=True)
        filtered = pf.filter(detections, frame_width, frame_height)
    """

    def __init__(self,
                 thresholds: Dict[str, float] = None,
                 enabled: bool = True):
        """
        Args:
            thresholds: Dict mapping label (lowercase) to minimum frame fraction.
                        Falls back to DEFAULT_MIN_FRAME_FRACTION for any label
                        not explicitly provided.
            enabled:    Master switch — if False, filter() is a no-op.
        """
        self.enabled = enabled
        # Merge caller-supplied overrides on top of the defaults
        self.thresholds = {**DEFAULT_MIN_FRAME_FRACTION, **(thresholds or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self,
               detections: List[Dict],
               frame_width: int,
               frame_height: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Split detections into (kept, discarded) based on size plausibility.

        Args:
            detections:   List of detection dicts with 'label' and 'bbox' keys.
            frame_width:  Width of the source frame in pixels.
            frame_height: Height of the source frame in pixels.

        Returns:
            kept:      Detections that passed the filter.
            discarded: Detections that were rejected (for logging purposes).
        """
        if not self.enabled:
            return detections, []

        frame_area = frame_width * frame_height
        if frame_area <= 0:
            return detections, []

        kept = []
        discarded = []

        for det in detections:
            label_lower = det.get('label', '').lower().strip()
            min_fraction = self._get_threshold(label_lower)

            if min_fraction is None:
                # No size constraint for this label — always keep
                kept.append(det)
                continue

            bbox = det.get('bbox', [0, 0, 0, 0])
            bbox_area = self._bbox_area(bbox)
            actual_fraction = bbox_area / frame_area

            if actual_fraction >= min_fraction:
                kept.append(det)
            else:
                discarded.append(det)
                logger.info(
                    f"[PLAUSIBILITY] Discarded '{det['label']}' "
                    f"(conf={det.get('confidence', 0):.2f}) — "
                    f"bbox covers {actual_fraction*100:.1f}% of frame, "
                    f"minimum required for this label: {min_fraction*100:.0f}%"
                )

        return kept, discarded

    def is_plausible(self,
                     label: str,
                     bbox: List[int],
                     frame_width: int,
                     frame_height: int) -> bool:
        """
        Convenience method to check a single detection.

        Returns True if the detection is plausible (should be kept).
        """
        kept, _ = self.filter(
            [{'label': label, 'bbox': bbox, 'confidence': 0.0}],
            frame_width, frame_height
        )
        return len(kept) > 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_threshold(self, label_lower: str):
        """
        Return the minimum frame fraction for a label, or None if
        there is no size constraint for that label.

        Checks exact match first, then substring match so that labels
        like "side-by-side refrigerator" still hit the "refrigerator" rule.
        """
        # Exact match
        if label_lower in self.thresholds:
            return self.thresholds[label_lower]

        # Substring match (e.g. "french door refrigerator" -> "refrigerator")
        for key, threshold in self.thresholds.items():
            if key in label_lower:
                return threshold

        return None

    @staticmethod
    def _bbox_area(bbox: List[int]) -> int:
        """Return pixel area of [x1, y1, x2, y2] bbox."""
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)