##############################################################################
# @Author: Peter Nolan
# @Contributor(s): Evan Rapoza, Joshua Tourigny, Max Manjos
# @Document: 'main.py'
#
# Description:
#   Main entry point for the Vision-Based Packing Project pipeline.
#
#   Orchestrates the full frame-by-frame processing loop:
#     1. Extract frames from the input video at a configurable interval.
#     2. Run two YOLO models (item detector + box detector) on each frame.
#     3. Apply a plausibility filter to remove physically-impossible detections.
#     4. Run ByteTrack multi-object tracking with MobileNetV2 re-ID.
#     5. Detect hand presence for exit-detection arming.
#     6. Scan for 1D / 2D barcodes to assign stable box IDs.
#     7. Detect item entry into boxes (EntryDetector).
#     8. Refine labels with LLaMA 3.2 Vision (FrameReasoner).
#     9. Register confirmed items in the ItemRegistry (deduplication).
#    10. Detect item removal (ExitDetector, two-stage human confirmation).
#    11. Write detection_log.csv, item_registry.json, refined_item_list.txt,
#        and an annotated output video.
#
#   All tunable parameters live in config.yaml — no code changes required
#   for typical deployment adjustments.
##############################################################################

from video_processor                import extractFrames
from detection.frame_loader         import loadFrames
from detection.object_detector      import ObjectDetector
from detection.frame_reasoner       import FrameReasoner
from detection.object_tracker       import ObjectTracker
from detection.video_annotator      import VideoAnnotator
from detection.entry_detector       import EntryDetector
from detection.exit_detector        import ExitDetector, confirmation_queue
from detection.plausibility_filter  import PlausibilityFilter
from detection.item_registry        import ItemRegistry
from detection.hand_detector        import HandDetector
from detection.barcode_scanner      import BarcodeScanner
from detection.box_tracker          import BoxTracker

import sys
import os
import csv
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List

# ── Force UTF-8 I/O on Windows to avoid charmap errors ──────────────────────
os.environ["PYTHONUTF8"]       = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# ── Allow imports from the project root (config_loader, etc.) ───────────────
sys.path.insert(0, '..')
from config_loader import get_config

import cv2

# ── Resolve the project root regardless of working directory ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── YOLO labels that are ambiguous or physically impossible inside a box ──────
# These trigger LLaMA refinement (uncertain_yolo = True) so the box context
# is deliberately withheld to prevent the model anchoring on wrong priors.
UNCERTAIN_YOLO_LABELS = {
    "bottle", "cell phone", "remote", "refrigerator",
    "microwave", "tv", "laptop", "cup", "vase", "bowl",
    "toothbrush", "hair drier", "scissors",
    "knife", "fork", "spoon",
    "bed", "couch", "sofa", "bench",
    "dining table", "desk", "chair",
    "sink", "toilet", "bathtub", "oven", "toaster",
    "clock", "potted plant", "fire hydrant",
}

# ── CSV column layout ─────────────────────────────────────────────────────────
CSV_HEADER = [
    "frame_index", "filename", "timestamp",
    "yolo_label", "confidence", "refined_label",
    "track_id", "instance_id", "box_id",
    "entry_detected", "exit_detected",
    "hand_detected", "is_uncertain", "plausibility_discarded",
]
INSTANCE_COL = CSV_HEADER.index("instance_id")
BOX_ID_COL   = CSV_HEADER.index("box_id")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: audio alert (Windows beep; falls back to terminal bell)
# ─────────────────────────────────────────────────────────────────────────────
def play_alert():
    """
    Play a brief audio alert when an exit event requires user confirmation.
    Uses winsound on Windows; falls back to a terminal bell character elsewhere.
    """
    try:
        import winsound
        winsound.Beep(1000, 300)
    except Exception:
        try:
            sys.stdout.write('\a')
            sys.stdout.flush()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Helper: logging setup
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(config):
    """
    Configure Python logging from config.yaml settings.

    Reads:
        logging.level        — e.g. "INFO", "DEBUG"
        logging.log_file     — path relative to project root (or None)
        logging.console_output — whether to echo to stdout

    Returns:
        A configured logger for main.py.
    """
    log_level      = getattr(logging, config.get('logging.level', 'INFO'))
    log_file       = config.get('logging.log_file', 'app.log')
    if log_file:
        log_file   = PROJECT_ROOT / log_file
    console_output = config.get('logging.console_output', True)

    handlers = []
    if console_output:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def is_box_label(label: str, box_labels: list) -> bool:
    """Return True if *label* matches any string in *box_labels* (case-insensitive)."""
    label_lower = label.lower()
    return any(bl.lower() in label_lower for bl in box_labels)


def get_bbox_area(bbox: list) -> int:
    """Return the pixel area of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def item_larger_than_box(item_bbox: list, box_bbox: list) -> bool:
    """
    Return True if the item's bounding box is larger than the box's bounding box.
    Used to skip physically-impossible detections (e.g. a sofa inside a small carton).
    """
    return get_bbox_area(item_bbox) > get_bbox_area(box_bbox)


def get_hand_bboxes(hand_detector: HandDetector, frame: np.ndarray) -> List[List[int]]:
    """
    Run the hand detection model on *frame* and return a list of
    bounding boxes [[x1,y1,x2,y2], ...] for every detected hand.

    Returns an empty list on failure (non-fatal; hand detection is
    used only for exit-detection arming, not for tracking).
    """
    try:
        results = hand_detector.model.predict(
            frame,
            conf=hand_detector.conf_threshold,
            verbose=False,
        )
        bboxes = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                bboxes.append(xyxy)
        return bboxes
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main(videoPath=None):
    """
    Run the full Vision-Based Packing Project pipeline on *videoPath*.

    If *videoPath* is None, falls back to video.default_video in config.yaml.
    Called by app.py (Flask UI) and can also be run directly:
        python src/main.py path/to/video.mp4
    """
    start_time = datetime.now()

    # Clear confirmation queue from any previous Flask run.
    # confirmation_queue is a module-level dict that persists while Flask is up,
    # so it must be flushed at the start of each new pipeline invocation.
    confirmation_queue.clear()

    config = get_config()
    logger = setup_logging(config)

    logger.info("=" * 60)
    logger.info("Vision-Based Packing Project — Pipeline Start")
    logger.info("=" * 60)

    # ── Directory setup ───────────────────────────────────────────────────────
    # framesDir    — extracted raw frames (JPEGs, one per time-step)
    # annotatedDir — YOLO cropped-item images saved per confirmed detection
    framesDir    = PROJECT_ROOT / config.get('paths.frames_dir',    'data/frames')
    annotatedDir = PROJECT_ROOT / config.get('paths.annotated_dir', 'data/yolo_frames')
    framesDir.mkdir(parents=True, exist_ok=True)
    annotatedDir.mkdir(parents=True, exist_ok=True)

    final_output_file  = PROJECT_ROOT / config.get('paths.refined_items_file', 'refined_item_list.txt')
    detection_log_file = PROJECT_ROOT / config.get('paths.detection_log_file', 'detection_log.csv')

    if videoPath is None:
        videoPath = PROJECT_ROOT / config.get('video.default_video', 'data/videos/packing_video.mp4')

    # ── Frame extraction ──────────────────────────────────────────────────────
    logger.info(f"Extracting frames from: {videoPath}")
    frame_interval = config.get('video.frame_interval', 1.0)
    frames_meta    = extractFrames(str(videoPath), str(framesDir), frame_interval)
    framesData     = loadFrames(str(framesDir))
    logger.info(f"Loaded {len(framesData)} frames for processing")

    # Build a filename → timestamp lookup from the extraction metadata
    meta_map = {}
    try:
        meta_map = {
            os.path.basename(m.get('path', '')): m.get('timestamp', None)
            for m in frames_meta
        }
    except Exception as e:
        logger.warning(f"Could not build timestamp map: {e}")

    # ── Object detector (dual YOLO) ───────────────────────────────────────────
    yolo_model         = PROJECT_ROOT / config.get('detection.yolo_model',  'models/yolo11l.pt')
    box_model_path     = PROJECT_ROOT / config.get('detection.box_model',   'models/cardboard_YOLO25.pt')
    conf_threshold     = config.get('detection.confidence_threshold',        0.50)
    box_conf_threshold = config.get('detection.box_confidence_threshold',    conf_threshold)
    box_labels         = config.get('detection.box_labels', ['box', 'carton', 'cardboard'])
    ignore_labels      = set(config.get('detection.ignore_labels', ['person']))
    use_tta            = config.get('detection.use_tta', False)

    detector = ObjectDetector(
        modelPath    = str(yolo_model),
        boxModelPath = str(box_model_path) if box_model_path.exists() else None,
        useTTA       = use_tta,
    )
    logger.info(f"ObjectDetector ready | conf={conf_threshold} | TTA={use_tta}")

    # ── Plausibility filter ───────────────────────────────────────────────────
    # Reads per-label minimum frame-fraction thresholds from config.yaml.
    # Detections whose bbox is too small for their label are discarded.
    plausibility_thresholds = config.get('plausibility_filter.thresholds', {})
    plausibility_enabled    = config.get('plausibility_filter.enabled', True)
    plausibility_filter     = PlausibilityFilter(custom_thresholds=plausibility_thresholds)
    logger.info(f"PlausibilityFilter: {'ENABLED' if plausibility_enabled else 'DISABLED'}")

    # ── LLaMA label reasoner ──────────────────────────────────────────────────
    llama_enabled = config.get('llama.enabled', True)
    reasoner      = None
    if llama_enabled:
        try:
            reasoner = FrameReasoner(
                model_name  = config.get('llama.model_name',  'llama3.2-vision'),
                max_retries = config.get('llama.max_retries', 3),
                retry_delay = config.get('llama.retry_delay', 1.0),
            )
            logger.info(f"FrameReasoner ready | model={config.get('llama.model_name')}")
        except Exception as e:
            logger.warning(f"LLaMA unavailable ({e}); falling back to YOLO labels only")
            reasoner = None

    # ── ByteTrack multi-object tracker ───────────────────────────────────────
    tracking_enabled = config.get('tracking.enabled', True)
    tracker          = None
    if tracking_enabled:
        tracker = ObjectTracker(
            track_thresh       = config.get('tracking.track_thresh',       0.5),
            track_buffer       = config.get('tracking.track_buffer',       60),
            match_thresh       = config.get('tracking.match_thresh',       0.7),
            second_match_thresh= config.get('tracking.second_match_thresh', 0.4),
            reid_enabled       = config.get('tracking.reid_enabled',       True),
            reid_thresh        = config.get('tracking.reid_thresh',        0.50),
            reid_buffer        = config.get('tracking.reid_buffer',        150),
        )
        logger.info("ObjectTracker (ByteTrack + MobileNetV2 re-ID) ready")

    # ── Hand detector ─────────────────────────────────────────────────────────
    hand_model_path = PROJECT_ROOT / config.get('detection.hand_model', 'models/hands_weight.pt')
    hand_detector   = None
    if hand_model_path.exists():
        try:
            hand_detector = HandDetector(
                model_path      = str(hand_model_path),
                conf_threshold  = config.get('detection.hand_conf_threshold', 0.35),
            )
            logger.info("HandDetector ready")
        except Exception as e:
            logger.warning(f"HandDetector unavailable ({e}); exit Stage 1 disabled")
    else:
        logger.warning(f"Hand model not found at {hand_model_path}; exit Stage 1 disabled")

    # ── Barcode / QR scanner ──────────────────────────────────────────────────
    barcode_scanner = BarcodeScanner(
        enabled             = config.get('barcode_scanner.enabled',             True),
        max_age             = config.get('barcode_scanner.max_age',             60),
        multi_scale         = config.get('barcode_scanner.multi_scale',         True),
        scale_factors       = config.get('barcode_scanner.scale_factors',       [1.0, 0.75, 0.5, 1.5]),
        proximity_threshold = config.get('barcode_scanner.proximity_threshold', 200),
    )

    # ── Box tracker (propagates barcode IDs across frames via IoU) ────────────
    box_tracker_enabled = config.get('box_tracker.enabled', True)
    box_tracker = BoxTracker(
        iou_threshold = config.get('box_tracker.iou_threshold', 0.30),
        max_age       = config.get('box_tracker.max_age',       30),
    ) if box_tracker_enabled else None
    logger.info(f"BoxTracker: {'ENABLED' if box_tracker_enabled else 'DISABLED'}")

    # ── Item registry (deduplication + stable instance IDs) ──────────────────
    registry = ItemRegistry(
        same_item_window           = config.get('item_registry.same_item_window',           20),
        label_similarity_threshold = config.get('item_registry.label_similarity_threshold', 0.5),
    )

    # ── Entry detector ────────────────────────────────────────────────────────
    entry_enabled  = config.get('entry_detection.enabled', True)
    entry_detector = None
    if entry_enabled:
        entry_detector = EntryDetector(
            entry_threshold = config.get('entry_detection.entry_threshold', 1),
            exit_threshold  = config.get('entry_detection.exit_threshold',  1),
            require_motion  = config.get('entry_detection.require_motion',  False),
        )

    # ── Exit detector ─────────────────────────────────────────────────────────
    exit_enabled  = config.get('exit_detection.enabled', True)
    exit_detector = None
    if exit_enabled:
        exit_detector = ExitDetector(
            absence_threshold      = config.get('exit_detection.absence_threshold',      5),
            hand_overlap_threshold = config.get('exit_detection.hand_overlap_threshold', 0.50),
            geometric_threshold    = config.get('exit_detection.geometric_threshold',    25),
        )

    # ── Video annotator ───────────────────────────────────────────────────────
    create_video   = config.get('output.create_video', True)
    annotated_path = PROJECT_ROOT / config.get(
        'paths.annotated_video_file', 'data/videos/output_annotated.mp4'
    )
    video_annotator = VideoAnnotator(
        output_path = str(annotated_path),
        fps         = config.get('output.video_fps',   10),
        codec       = config.get('output.video_codec', 'mp4v'),
    ) if create_video else None

    # ─────────────────────────────────────────────────────────────────────────
    # Per-run state
    # ─────────────────────────────────────────────────────────────────────────
    # label_cache: track_id → refined_label
    #   Populated on the first LLaMA call for a track; used to skip re-calling
    #   LLaMA on subsequent frames for the same tracked object (fast path).
    label_cache: dict = {}

    # Records which (track_id, box_id) pairs have been confirmed as entries.
    confirmed_entries: set = set()

    # Tracks which box IDs have been "completed" (camera moved to next box).
    completed_box_ids: set = set()

    # Tracks the previous barcode-resolved box ID for change detection.
    prev_box_id: str = 'BOX-001'

    # Pipeline-wide detection counters for the summary log.
    total_detections          = 0
    total_plausibility_discards = 0

    # Open the CSV log for writing.
    csv_rows: list = []

    # ─────────────────────────────────────────────────────────────────────────
    # Per-frame processing loop
    # ─────────────────────────────────────────────────────────────────────────
    for f in framesData:
        frame_idx    = f.get('index', 0)
        filename     = f.get('filename', '')
        originalFrame = f.get('frame')
        timestamp    = meta_map.get(filename, frame_idx)

        if originalFrame is None:
            logger.warning(f"Skipping frame {frame_idx}: no image data")
            continue

        frame_h, frame_w = originalFrame.shape[:2]

        # ── Barcode scan ─────────────────────────────────────────────────────
        barcode_scanner.scan_frame(originalFrame, frame_idx)

        # Resolve box ID from the most-recently-seen barcode, or fall back.
        FALLBACK_IDS   = {'BOX-001', 'box-001'}
        active_codes   = barcode_scanner.get_active_codes(frame_idx)
        current_box_id = (
            max(active_codes.items(), key=lambda x: x[1]['last_seen_frame'])[0]
            if active_codes else 'BOX-001'
        )

        # ── Box-ID change detection ───────────────────────────────────────────
        # When the barcode switches (camera moved to a new box), flush all
        # per-box state so items are re-evaluated under the new box context.
        if (current_box_id not in FALLBACK_IDS
                and prev_box_id not in FALLBACK_IDS
                and current_box_id != prev_box_id):
            logger.info(f"[BOX-CHANGE] {prev_box_id!r} → {current_box_id!r} at frame {frame_idx}")
            label_cache.clear()
            completed_box_ids.add(prev_box_id)
            if entry_detector:
                entry_detector.confirmed_entries.clear()
                entry_detector.entered_items.clear()
                entry_detector.inside_counter.clear()
                entry_detector.outside_counter.clear()
                entry_detector.track_history.clear()
                entry_detector._last_seen_frame.clear()

        prev_box_id = current_box_id

        # ── YOLO detection ────────────────────────────────────────────────────
        detections_raw = detector.detectObjects(
            originalFrame,
            confThresh    = conf_threshold,
            boxConfThresh = box_conf_threshold,
        )

        # Separate box detections from item detections
        box_detections_raw = [
            d for d in detections_raw if is_box_label(d['label'], box_labels)
        ]
        item_detections = [
            d for d in detections_raw if not is_box_label(d['label'], box_labels)
        ]

        # Resolve box IDs via BoxTracker (IoU match + barcode propagation)
        if box_tracker:
            box_detections = box_tracker.update(box_detections_raw, current_box_id, frame_idx)
        else:
            box_detections = [
                {**d, 'box_id': current_box_id} for d in box_detections_raw
            ]

        # ── Hand detection ────────────────────────────────────────────────────
        hand_detected = False
        hand_bboxes   = []
        if hand_detector:
            hand_detected = hand_detector.detect(originalFrame)
            if hand_detected:
                hand_bboxes = get_hand_bboxes(hand_detector, originalFrame)

        # ── Plausibility filter (pass 1 — on raw YOLO labels) ────────────────
        if plausibility_enabled:
            item_detections, _ = plausibility_filter.filter(
                item_detections, frame_w, frame_h
            )

        # ── ByteTrack update ──────────────────────────────────────────────────
        tracked = []
        if tracker:
            tracked = tracker.update(item_detections, originalFrame)
        else:
            # Without a tracker every detection gets a synthetic track_id of -1
            tracked = [
                {**d, 'track_id': -1} for d in item_detections
            ]

        total_detections += len(tracked)

        # ── Exit detector sync (must run before per-object loop) ──────────────
        if exit_detector:
            exit_detector.sync_registry(registry)

            # Stage 1: arm candidates overlapping a detected hand
            active_track_bboxes = {
                obj['track_id']: obj['bbox']
                for obj in tracked
                if obj.get('track_id') is not None
            }
            exit_detector.update_hand_flags(
                active_track_bboxes, hand_bboxes, frame_idx, hand_detected
            )

            # Stage 2: count absences; queue confirmation requests
            active_track_ids = set(active_track_bboxes.keys())
            confirmed_removals = exit_detector.check_absences(
                active_track_ids, frame_idx, float(timestamp or 0), originalFrame
            )
            for instance_id in confirmed_removals:
                registry.mark_removed(instance_id, frame_idx, float(timestamp or 0))
                play_alert()
                logger.info(f"[EXIT CONFIRMED] instance #{instance_id} removed at frame {frame_idx}")

        # ── Per-detection processing ──────────────────────────────────────────
        for obj in tracked:
            yoloLabel  = obj.get('label', 'unknown')
            confidence = obj.get('confidence', 0.0)
            bbox       = obj.get('bbox', [0, 0, 0, 0])
            track_id   = obj.get('track_id')

            # Skip labels that are always-ignored (people, etc.)
            if yoloLabel.lower() in ignore_labels:
                continue

            # Labels too large to fit in a packing box
            furniture_labels = {'bed', 'couch', 'sofa', 'desk', 'dining table'}

            # ── Fast path: label already cached for this track ────────────────
            # Skip LLaMA; re-run entry detection to keep the registry current.
            if track_id is not None and track_id in label_cache:
                refined_label  = label_cache[track_id]
                entry_detected = False
                box_id         = None

                if entry_detector and box_detections:
                    for box_det in box_detections:
                        box_bbox = box_det['bbox']
                        if item_larger_than_box(bbox, box_bbox):
                            continue
                        confirmed = entry_detector.detect_entry(
                            track_id          = track_id,
                            item_bbox         = bbox,
                            box_id            = box_det['box_id'],
                            box_bbox          = box_bbox,
                            frame_number      = frame_idx,
                            overlap_threshold = config.get('entry_detection.overlap_threshold', 0.20),
                        )
                        if confirmed:
                            entry_detected = True
                            box_id         = box_det['box_id']
                            logger.info(
                                f"[ENTRY] Frame {frame_idx}: Track#{track_id} "
                                f"'{refined_label}' → {box_id}"
                            )
                            break

                # Skip furniture that somehow entered detection (size guard missed it)
                if refined_label.lower().strip() in furniture_labels and entry_detected and box_id:
                    csv_rows.append([
                        frame_idx, filename, timestamp, yoloLabel, confidence,
                        refined_label, track_id, None, None,
                        False, False, hand_detected, False, None,
                    ])
                    continue

                if entry_enabled and not entry_detected:
                    csv_rows.append([
                        frame_idx, filename, timestamp, yoloLabel, confidence,
                        None, track_id, None, None,
                        False, False, hand_detected, False, None,
                    ])
                    continue

                # Notify the entry detector that this track is still alive this frame
                if entry_detector:
                    entry_detector.notify_track_seen(track_id, frame_idx)

                # Update the registry with this continued detection
                instance_id, _ = registry.register_entry(
                    track_id      = track_id,
                    refined_label = refined_label,
                    box_id        = box_id or current_box_id,
                    frame         = frame_idx,
                    timestamp     = float(timestamp or 0),
                    yolo_label    = yoloLabel,
                )

                csv_rows.append([
                    frame_idx, filename, timestamp, yoloLabel, confidence,
                    refined_label, track_id, instance_id, box_id,
                    entry_detected, False, hand_detected, False, None,
                ])
                continue

            # ── Slow path: new track — run entry detection + LLaMA ───────────
            entry_detected = False
            box_id         = None

            if entry_detector and box_detections:
                for box_det in box_detections:
                    box_bbox = box_det['bbox']
                    if item_larger_than_box(bbox, box_bbox):
                        continue
                    confirmed = entry_detector.detect_entry(
                        track_id          = track_id,
                        item_bbox         = bbox,
                        box_id            = box_det['box_id'],
                        box_bbox          = box_bbox,
                        frame_number      = frame_idx,
                        overlap_threshold = config.get('entry_detection.overlap_threshold', 0.20),
                    )
                    if confirmed:
                        entry_detected = True
                        box_id         = box_det['box_id']
                        break

            # Item not yet inside any box — log and move on
            if entry_enabled and not entry_detected:
                csv_rows.append([
                    frame_idx, filename, timestamp, yoloLabel, confidence,
                    None, track_id, None, None,
                    False, False, hand_detected, False, None,
                ])
                continue

            # ── Crop the item from the full frame ─────────────────────────────
            x1, y1, x2, y2 = bbox
            crop = originalFrame[y1:y2, x1:x2]

            if crop.size == 0:
                logger.warning(f"Empty crop for '{yoloLabel}' at frame {frame_idx}")
                refined_label = yoloLabel
            else:
                # ── LLaMA label refinement ────────────────────────────────────
                # Pass box_context only for certain-YOLO labels; withhold it for
                # ambiguous labels to prevent the model anchoring on wrong priors.
                uncertain_yolo   = yoloLabel.lower().strip() in UNCERTAIN_YOLO_LABELS
                box_context_full = registry.get_unique_labels_for_box(box_id or "GLOBAL")
                box_context      = None if uncertain_yolo else box_context_full

                if reasoner:
                    refined_label = reasoner.refineDetection(
                        crop, yoloLabel, box_context=box_context
                    )
                else:
                    refined_label = yoloLabel

                refined_label = refined_label or yoloLabel

                # Attempt label normalization (e.g. "Water Bottle" → "Bottle")
                if reasoner and refined_label.lower().strip() == yoloLabel.lower().strip():
                    original_label = refined_label
                    refined_label  = reasoner.normalize_label(refined_label)
                    if original_label != refined_label:
                        logger.info(f"[NORMALIZE] '{original_label}' → '{refined_label}'")

                # ── Plausibility filter (pass 2 — on refined label) ───────────
                # A second pass catches cases where LLaMA returns a large-appliance
                # label from a small crop (e.g. "Refrigerator" for a tiny image).
                if plausibility_enabled:
                    _, discards = plausibility_filter.filter(
                        [{'label': refined_label, 'confidence': confidence, 'bbox': bbox}],
                        frame_w, frame_h,
                    )
                    if discards:
                        total_plausibility_discards += 1
                        csv_rows.append([
                            frame_idx, filename, timestamp, yoloLabel, confidence,
                            refined_label, track_id, None, None,
                            False, False, hand_detected, False, True,
                        ])
                        continue

                # Skip furniture even if it passed the plausibility filter
                if refined_label.lower().strip() in furniture_labels and box_id:
                    total_plausibility_discards += 1
                    csv_rows.append([
                        frame_idx, filename, timestamp, yoloLabel, confidence,
                        refined_label, track_id, None, None,
                        False, False, hand_detected, False, True,
                    ])
                    continue

                # ── Save cropped image to yolo_frames ─────────────────────────
                # BUG FIX: This block was previously outside the slow-path else
                # branch, meaning `crop` was None on fast-path frames (cached
                # tracks) and the save silently failed every other frame.
                # It now lives here, immediately after the crop is validated.
                if config.get('output.save_cropped_images', True):
                    safe_label = (refined_label or yoloLabel).replace(' ', '_').replace('/', '_')
                    crop_path  = annotatedDir / f"frame{frame_idx:05d}_track{track_id}_{safe_label}.jpg"
                    success    = cv2.imwrite(str(crop_path), crop)
                    if not success:
                        logger.warning(f"Failed to write crop: {crop_path}")

            # Cache the refined label so fast path skips LLaMA on future frames
            if track_id is not None:
                label_cache[track_id] = refined_label

            # ── Register in ItemRegistry ──────────────────────────────────────
            is_uncertain = yoloLabel.lower().strip() in UNCERTAIN_YOLO_LABELS
            instance_id, is_new = registry.register_entry(
                track_id      = track_id,
                refined_label = refined_label,
                box_id        = box_id or current_box_id,
                frame         = frame_idx,
                timestamp     = float(timestamp or 0),
                yolo_label    = yoloLabel,
                is_uncertain  = is_uncertain,
            )

            if is_new:
                logger.info(
                    f"[NEW ITEM] #{instance_id} '{refined_label}' "
                    f"(YOLO: '{yoloLabel}') in {box_id} at frame {frame_idx}"
                )

            csv_rows.append([
                frame_idx, filename, timestamp, yoloLabel, confidence,
                refined_label, track_id, instance_id, box_id,
                entry_detected, False, hand_detected, is_uncertain, None,
            ])

        # ── Annotated video frame ─────────────────────────────────────────────
        if video_annotator and create_video:
            video_annotator.add_frame(
                originalFrame, tracked, frame_idx, timestamp, hand_detected
            )

    # ── End of frame loop ─────────────────────────────────────────────────────
    logger.info("Frame loop complete. Writing outputs...")

    # Finalize annotated video
    if video_annotator:
        video_annotator.finalize()

    # ── Back-fill box IDs in CSV rows from the registry ──────────────────────
    # After the loop completes, the registry has the authoritative box_id for
    # each instance. Rows written during the loop may have used a placeholder;
    # this pass corrects them using the registry and the best barcode scan.
    all_scanned  = barcode_scanner.summary().get('codes', {})
    best_barcode = None
    if all_scanned:
        candidates = [
            (data, code)
            for code, data in all_scanned.items()
            if code not in FALLBACK_IDS
        ]
        if candidates:
            best_barcode = min(candidates, key=lambda x: x[0]['first_frame'])[1]

    instance_to_box = {
        inst.instance_id: inst.box_id
        for inst in registry.get_all_items()
    }

    for row in csv_rows:
        raw_iid     = row[INSTANCE_COL]
        instance_id = int(raw_iid) if raw_iid not in (None, '', 'None') else None
        reg_id      = instance_to_box.get(instance_id) if instance_id is not None else None
        if reg_id and str(reg_id) not in FALLBACK_IDS:
            row[BOX_ID_COL] = reg_id
        elif best_barcode and str(row[BOX_ID_COL]) in FALLBACK_IDS:
            row[BOX_ID_COL] = best_barcode

    # ── Write detection_log.csv ───────────────────────────────────────────────
    with open(detection_log_file, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(CSV_HEADER)
        writer.writerows(csv_rows)
    logger.info(f"Detection log → {detection_log_file}")

    # ── Write item_registry.json ──────────────────────────────────────────────
    registry_data = registry.export_to_dict()
    registry_path = PROJECT_ROOT / 'item_registry.json'
    with open(registry_path, 'w', encoding='utf-8') as jf:
        json.dump(registry_data, jf, indent=2)
    logger.info(f"Item registry → {registry_path}")

    # ── Write box_mappings.json ───────────────────────────────────────────────
    box_map_path = PROJECT_ROOT / 'box_mappings.json'
    with open(box_map_path, 'w', encoding='utf-8') as jf:
        json.dump(registry_data.get('by_box', {}), jf, indent=2)
    logger.info(f"Box mappings → {box_map_path}")

    # ── Write refined_item_list.txt ───────────────────────────────────────────
    with open(final_output_file, 'w', encoding='utf-8') as tf:
        items_by_box = registry_data.get('by_box', {})
        for box_id, items in items_by_box.items():
            tf.write(f"[{box_id}]\n")
            for item in items:
                status = " (removed)" if item.get('status') == 'removed' else ""
                tf.write(f"  - {item['label']}{status}\n")
    logger.info(f"Item list → {final_output_file}")

    # ── Pipeline summary ──────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    summary = registry_data.get('summary', {})
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Total detections   : {total_detections}")
    logger.info(f"  Plausibility drops : {total_plausibility_discards}")
    logger.info(f"  Unique items found : {summary.get('total_unique_items', 0)}")
    logger.info(f"  Items in box       : {summary.get('items_in_box', 0)}")
    logger.info(f"  Items removed      : {summary.get('items_removed', 0)}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Optionally pass a video path as the first CLI argument.
    # If omitted, falls back to video.default_video in config.yaml.
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(video_path)