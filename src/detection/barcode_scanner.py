##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'barcode_scanner.py'
#
# Description:
# Detects and decodes both 1D barcodes AND QR codes in video frames
# to assign unique IDs to packing boxes.
#
# Replaces the placeholder 'BOX-001' logic in main.py.
# Integrates with BoxTracker to persist barcode-assigned IDs
# across frames where the code is not directly visible.
#
# Supports pyzbar for decoding; falls back gracefully if unavailable.
##############################################

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Attempt to import pyzbar ──────────────────────────────────────────────────
try:
    from pyzbar import pyzbar as _pyzbar
    _PYZBAR_AVAILABLE = True
    logger.debug("pyzbar loaded successfully.")
except ImportError:
    _PYZBAR_AVAILABLE = False
    logger.warning(
        "pyzbar is not installed. Barcode/QR scanning will be DISABLED. "
        "Install with: pip install pyzbar"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_variants(frame: np.ndarray) -> List[np.ndarray]:
    """
    Return a list of grayscale preprocessed variants of the frame.
    Trying multiple variants significantly improves decode rates on
    blurry, low-contrast, or partially occluded codes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variants = [gray]

    # CLAHE – equalises local contrast, good for uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))

    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    variants.append(sharpened)

    # Adaptive threshold – handles barcode on textured/coloured surfaces
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    variants.append(thresh)

    return variants


def _decode_frame(gray: np.ndarray) -> List[Dict]:
    """
    Run pyzbar on a single grayscale image.
    Returns a list of decoded symbol dicts.
    """
    if not _PYZBAR_AVAILABLE:
        return []
    try:
        symbols = _pyzbar.decode(gray)
    except Exception as exc:
        logger.debug(f"pyzbar decode error: {exc}")
        return []

    results = []
    for sym in symbols:
        try:
            data = sym.data.decode("utf-8").strip()
        except (UnicodeDecodeError, AttributeError):
            data = sym.data.hex()

        x, y, w, h = sym.rect
        bbox = [x, y, x + w, y + h]
        center = (x + w // 2, y + h // 2)
        points = [(p.x, p.y) for p in sym.polygon]

        results.append({
            "data":    data,
            "type":    sym.type,          # e.g. "QRCODE", "CODE128", "EAN13" …
            "bbox":    bbox,
            "center":  center,
            "polygon": points,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# BarcodeScanner
# ─────────────────────────────────────────────────────────────────────────────

class BarcodeScanner:
    """
    Scans video frames for both 1D barcodes and QR codes attached to boxes.

    Typical usage inside the main frame loop
    ─────────────────────────────────────────
        scanner = BarcodeScanner(enabled=True)

        for frame_data in frames:
            frame     = frame_data["frame"]
            frame_idx = frame_data["index"]

            scanned = scanner.scan(frame, frame_idx)
            scanner.update_registry(scanned, frame_idx)

        # Later – resolve a YOLO box detection to a barcode ID:
        box_id = scanner.resolve_box_id(
            box_bbox  = det["bbox"],
            frame_idx = frame_idx,
        )
    """

    def __init__(
        self,
        enabled:            bool = True,
        min_confidence:     float = 0.0,    # pyzbar doesn't give confidence; reserved
        max_age:            int   = 60,     # frames a code remains "active" without re-scan
        multi_scale:        bool  = True,   # try downscaled copies for distant codes
        scale_factors:      Optional[List[float]] = None,
        proximity_threshold: int  = 200,    # pixels – max distance for bbox association
    ):
        self.enabled             = enabled and _PYZBAR_AVAILABLE
        self.max_age             = max_age
        self.multi_scale         = multi_scale
        self.scale_factors       = scale_factors or [1.0, 0.75, 0.5, 1.5]
        self.proximity_threshold = proximity_threshold

        # code_data → {bbox, center, last_seen_frame, type, scan_count}
        self._registry: Dict[str, Dict] = {}

        if not self.enabled:
            reason = "disabled by config" if not enabled else "pyzbar unavailable"
            logger.info(f"BarcodeScanner: DISABLED ({reason})")
        else:
            logger.info(
                f"BarcodeScanner: ENABLED  "
                f"(max_age={max_age}, multi_scale={multi_scale}, "
                f"scales={self.scale_factors}, proximity={proximity_threshold}px)"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def scan(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """
        Scan *frame* for barcodes/QR codes.

        Returns a deduplicated list of detected symbols:
            [{'data': str, 'type': str, 'bbox': [...], 'center': (...), 'polygon': [...]}]

        Updates the internal registry as a side-effect.
        """
        if not self.enabled:
            return []

        found: Dict[str, Dict] = {}   # data → best hit (dedup by data value)

        variants = _preprocess_variants(frame)
        h, w = frame.shape[:2]

        # --- Base resolution variants ---
        for variant in variants:
            for sym in _decode_frame(variant):
                if sym["data"] not in found:
                    found[sym["data"]] = sym

        # --- Multi-scale pass ---
        if self.multi_scale:
            gray_base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for scale in self.scale_factors:
                if scale == 1.0:
                    continue
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                scaled = cv2.resize(gray_base, (new_w, new_h))
                for sym in _decode_frame(scaled):
                    if sym["data"] in found:
                        continue
                    # Re-map bbox/center back to original resolution
                    inv = 1.0 / scale
                    x1, y1, x2, y2 = sym["bbox"]
                    sym["bbox"]   = [int(x1*inv), int(y1*inv),
                                     int(x2*inv), int(y2*inv)]
                    sym["center"] = (int(sym["center"][0]*inv),
                                     int(sym["center"][1]*inv))
                    sym["polygon"] = [(int(p[0]*inv), int(p[1]*inv))
                                      for p in sym["polygon"]]
                    found[sym["data"]] = sym

        results = list(found.values())

        # Update registry
        self.update_registry(results, frame_idx)

        if results:
            logger.debug(
                f"Frame {frame_idx}: scanned {len(results)} code(s) → "
                + ", ".join(f"{s['data']} ({s['type']})" for s in results)
            )

        return results

    def update_registry(self, scanned: List[Dict], frame_idx: int) -> None:
        """Merge a list of freshly scanned symbols into the persistent registry."""
        for sym in scanned:
            data = sym["data"]
            if data in self._registry:
                entry = self._registry[data]
                entry["bbox"]            = sym["bbox"]
                entry["center"]          = sym["center"]
                entry["last_seen_frame"] = frame_idx
                entry["scan_count"]     += 1
            else:
                self._registry[data] = {
                    "bbox":            sym["bbox"],
                    "center":          sym["center"],
                    "type":            sym["type"],
                    "last_seen_frame": frame_idx,
                    "first_seen_frame": frame_idx,
                    "scan_count":      1,
                }
                logger.info(
                    f"[BARCODE] New code registered: '{data}' "
                    f"(type={sym['type']}, frame={frame_idx})"
                )

    def get_active_codes(self, current_frame: int) -> Dict[str, Dict]:
        """
        Return all codes seen within *max_age* frames of *current_frame*.
        """
        return {
            data: info
            for data, info in self._registry.items()
            if (current_frame - info["last_seen_frame"]) <= self.max_age
        }

    def resolve_box_id(
        self,
        box_bbox:   List[int],
        frame_idx:  int,
    ) -> Optional[str]:
        """
        Given a YOLO box detection bounding box, return the barcode/QR value
        of the code that is spatially closest to (and within) the box.

        Returns None if no active code is near enough.

        Strategy
        ────────
        1. Prefer codes whose bounding box *overlaps* the box detection bbox.
        2. Fall back to proximity (centre-to-centre distance).
        """
        active = self.get_active_codes(frame_idx)
        if not active:
            return None

        bx1, by1, bx2, by2 = box_bbox
        box_cx = (bx1 + bx2) / 2
        box_cy = (by1 + by2) / 2

        best_id:   Optional[str]   = None
        best_score: float          = float("inf")

        for data, info in active.items():
            cx, cy = info["center"]
            cx1, cy1, cx2, cy2 = info["bbox"]

            # Check overlap first (barcode stuck *on* the box)
            overlap_x = max(0, min(bx2, cx2) - max(bx1, cx1))
            overlap_y = max(0, min(by2, cy2) - max(by1, cy1))
            if overlap_x > 0 and overlap_y > 0:
                # Overlapping → use distance as tiebreaker but strongly prefer
                dist = 0.0  # treat overlap as zero distance
            else:
                dist = np.sqrt((cx - box_cx)**2 + (cy - box_cy)**2)
                if dist > self.proximity_threshold:
                    continue

            if dist < best_score:
                best_score = dist
                best_id    = data

        return best_id

    def annotate_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Draw all active barcodes/QR codes onto *frame* for debugging/video output.
        Returns a copy with annotations drawn.
        """
        annotated = frame.copy()
        active = self.get_active_codes(frame_idx)

        for data, info in active.items():
            bbox = info["bbox"]
            age  = frame_idx - info["last_seen_frame"]

            # Colour fades from bright green (fresh) to orange (stale)
            freshness = max(0.0, 1.0 - age / max(1, self.max_age))
            g = int(255 * freshness)
            r = int(255 * (1.0 - freshness))
            colour = (0, g, r)

            # Bounding box
            cv2.rectangle(
                annotated,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                colour, 2,
            )

            # Label background + text
            label       = f"{data} ({info['type']})"
            font        = cv2.FONT_HERSHEY_SIMPLEX
            font_scale  = 0.55
            thickness   = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            top_y = max(bbox[1] - th - baseline - 6, 0)
            cv2.rectangle(
                annotated,
                (bbox[0], top_y),
                (bbox[0] + tw + 4, bbox[1]),
                colour, -1,
            )
            cv2.putText(
                annotated, label,
                (bbox[0] + 2, bbox[1] - baseline - 2),
                font, font_scale, (0, 0, 0), thickness,
            )

        return annotated

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return a summary dict of all codes ever seen."""
        return {
            "total_unique_codes": len(self._registry),
            "codes": {
                data: {
                    "type":        info["type"],
                    "first_frame": info["first_seen_frame"],
                    "last_frame":  info["last_seen_frame"],
                    "scan_count":  info["scan_count"],
                }
                for data, info in self._registry.items()
            },
        }