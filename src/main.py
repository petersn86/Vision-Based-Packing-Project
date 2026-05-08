##############################################################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'main.py'
#
# Description:
# Main entry point for the Vision-Based-Packing-Project pipeline.
###############################################################################

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
import sys, os
import time

os.environ["PYTHONUTF8"]       = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, '..')
from config_loader import get_config
import cv2
from pathlib import Path
import csv
import json
import logging
from datetime import datetime
import numpy as np
from typing import List

# ------------------ Project Root ------------------ #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# YOLO labels too generic or impossible to trust inside a packing box.
UNCERTAIN_YOLO_LABELS = {
    "bottle", "refrigerator",
    "microwave", "tv", "laptop", "cup", "vase", "bowl",
     "hair drier", "fork", "spoon",
    "bed", "couch", "sofa", "bench",
    "dining table", "desk", "chair",
    "sink", "toilet", "bathtub", "oven", "toaster",
    "clock", "potted plant", "fire hydrant",
}


# ------------------ Audio Alert ------------------ #
def play_alert():
    try:
        import winsound
        winsound.Beep(1000, 300)
    except Exception:
        try:
            sys.stdout.write('\a')
            sys.stdout.flush()
        except Exception:
            pass


# ------------------ Logging Setup ----------------- #
def setup_logging(config):
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
        force=True
    )
    return logging.getLogger(__name__)


# ------------------ Helper Functions ------------------ #
def is_box_label(label: str, box_labels: list) -> bool:
    label_lower = label.lower()
    for box_label in box_labels:
        if box_label.lower() in label_lower:
            return True
    return False


def get_bbox_area(bbox: list) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def item_larger_than_box(item_bbox: list, box_bbox: list) -> bool:
    return get_bbox_area(item_bbox) > get_bbox_area(box_bbox)


def get_hand_bboxes(hand_detector: HandDetector, frame: np.ndarray) -> List[List[int]]:
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


# ------------------ Main Pipeline ----------------- #
def main(videoPath=None):
    start_time = datetime.now()

    # Clear confirmation queue from any previous run — it is a module-level
    # dict that persists in memory between pipeline calls when Flask stays up.
    confirmation_queue.clear()

    config = get_config()
    logger = setup_logging(config)

    logger.info("="*60)
    logger.info("Vision-Based Packing Project - YOLO Box Entry Detection")
    logger.info("="*60)

    # ---------------- Paths ---------------- #
    framesDir    = PROJECT_ROOT / config.get('paths.frames_dir',    'data/frames')
    annotatedDir = PROJECT_ROOT / config.get('paths.annotated_dir', 'data/yolo_frames')
    framesDir.mkdir(parents=True, exist_ok=True)
    annotatedDir.mkdir(parents=True, exist_ok=True)

    final_output_file  = PROJECT_ROOT / config.get('paths.refined_items_file', 'refined_item_list.txt')
    detection_log_file = PROJECT_ROOT / config.get('paths.detection_log_file', 'detection_log.csv')

    if videoPath is None:
        videoPath = PROJECT_ROOT / config.get('video.default_video', 'data/videos/packing_video.mp4')

    # ---------------- Frame Extraction ---------------- #
    logger.info(f"Extracting frames from video: {videoPath}")
    frame_interval = config.get('video.frame_interval', 2.0)
    frames_meta    = extractFrames(str(videoPath), str(framesDir), frame_interval)
    framesData     = loadFrames(str(framesDir))
    logger.info(f"Loaded {len(framesData)} frames for processing")

    meta_map = {}
    try:
        meta_map = {
            os.path.basename(m.get('path', '')): m.get('timestamp', None)
            for m in frames_meta
        }
    except Exception as e:
        logger.warning(f"Could not build timestamp mapping: {e}")

    # ---------------- Object Detector ---------------- #
    yolo_model         = PROJECT_ROOT / config.get('detection.yolo_model',  'models/yolo11l.pt')
    box_model_path     = PROJECT_ROOT / config.get('detection.box_model',   'models/cardboard_YOLO25.pt')
    conf_threshold     = config.get('detection.confidence_threshold',       0.35)
    box_conf_threshold = config.get('detection.box_confidence_threshold',   conf_threshold)
    box_labels         = config.get('detection.box_labels', ['box', 'carton', 'cardboard'])

    logger.info(f"Initializing YOLO models: {yolo_model}, {box_model_path}")
    detector = ObjectDetector(str(yolo_model), str(box_model_path))

    # ---------------- Hand Detector ---------------- #
    hand_model_path = PROJECT_ROOT / config.get('detection.hand_model', 'models/hands_weights.pt')
    hand_conf       = config.get('detection.hand_confidence_threshold', conf_threshold)
    hand_detector   = HandDetector(
        model_path=str(hand_model_path),
        conf_threshold=hand_conf,
    )

    # ---------------- Plausibility Filter ---------------- #
    plausibility_filter = PlausibilityFilter(
        thresholds=config.get('plausibility_filter.thresholds', {}),
        enabled=config.get('plausibility_filter.enabled', True)
    )
    logger.info(f"Plausibility filter: {'ENABLED' if plausibility_filter.enabled else 'DISABLED'}")

    # ---------------- Item Tracker ---------------- #
    tracking_enabled = config.get('tracking.enabled', True)
    tracker = None
    if tracking_enabled:
        tracker = ObjectTracker(
            track_thresh=config.get('tracking.track_thresh',               0.5),
            track_buffer=config.get('tracking.track_buffer',               60),
            match_thresh=config.get('tracking.match_thresh',               0.7),
            second_match_thresh=config.get('tracking.second_match_thresh', 0.4),
            reid_enabled=config.get('tracking.reid_enabled',               True),
            reid_thresh=config.get('tracking.reid_thresh',                 0.50),
            reid_buffer=config.get('tracking.reid_buffer',                 150),
        )
        logger.info("Item tracking enabled (ByteTrack + MobileNetV2 re-ID)")
    else:
        logger.info("Item tracking disabled")

    # ---------------- Barcode Scanner ---------------- #
    barcode_scanner = BarcodeScanner(
        enabled             = config.get('barcode_scanner.enabled',             True),
        max_age             = config.get('barcode_scanner.max_age',             60),
        multi_scale         = config.get('barcode_scanner.multi_scale',         True),
        scale_factors       = config.get('barcode_scanner.scale_factors',       [1.0, 0.75, 0.5, 1.5]),
        proximity_threshold = config.get('barcode_scanner.proximity_threshold', 200),
    )

    # ---------------- Box Tracker ---------------- #
    box_tracker_enabled = config.get('box_tracker.enabled', True)
    box_tracker = BoxTracker(
        iou_threshold = config.get('box_tracker.iou_threshold', 0.30),
        max_age       = config.get('box_tracker.max_age',       30),
    ) if box_tracker_enabled else None
    logger.info(f"BoxTracker: {'ENABLED' if box_tracker_enabled else 'DISABLED'}")

    # ---------------- Item Registry ---------------- #
    same_item_window           = config.get('item_registry.same_item_window', 20)
    label_similarity_threshold = config.get('item_registry.label_similarity_threshold', 0.5)
    registry = ItemRegistry(
        same_item_window=same_item_window,
        label_similarity_threshold=label_similarity_threshold
    )
    logger.info(
        f"ItemRegistry initialised (same_item_window={same_item_window} frames, "
        f"label_similarity_threshold={label_similarity_threshold})"
    )

    # ---------------- Entry Detector ---------------- #
    entry_enabled  = config.get('entry_detection.enabled', True)
    entry_detector = None
    if entry_enabled:
        entry_detector = EntryDetector(
            entry_threshold=config.get('entry_detection.entry_threshold', 3),
            exit_threshold =config.get('entry_detection.exit_threshold',  5),
            require_motion =config.get('entry_detection.require_motion',  False),
        )
        logger.info(
            f"Entry detection enabled | "
            f"threshold={config.get('entry_detection.entry_threshold', 3)} | "
            f"overlap={config.get('entry_detection.overlap_threshold', 0.5)} | "
            f"require_motion={config.get('entry_detection.require_motion', False)}"
        )

    # ---------------- Exit Detector ---------------- #
    exit_detection_enabled = config.get('exit_detection.enabled', True)
    exit_detector = None
    if exit_detection_enabled:
        exit_detector = ExitDetector(
            absence_threshold      = config.get('exit_detection.absence_threshold',      8),
            hand_overlap_threshold = config.get('exit_detection.hand_overlap_threshold', 0.30),
            geometric_threshold    = config.get('exit_detection.geometric_threshold',    15),
        )
        logger.info(
            f"Exit detection enabled (human confirmation + audio alert) | "
            f"absence={config.get('exit_detection.absence_threshold', 8)} frames | "
            f"hand_overlap={config.get('exit_detection.hand_overlap_threshold', 0.30)}"
        )
    else:
        logger.info("Exit detection disabled")

    # ---------------- LLaMA Reasoner (label refinement only) ---------------- #
    llama_enabled = config.get('llama.enabled', True)
    reasoner      = None
    if llama_enabled:
        llama_model = config.get('llama.model_name', 'llama3.2-vision')
        logger.info(f"Initializing LLaMA reasoner: {llama_model}")
        reasoner = FrameReasoner(model_name=llama_model)

    # ---------------- Video Annotator ---------------- #
    video_output_enabled = config.get('output.create_video', True)
    annotator            = None
    detections_per_frame = {}
    if video_output_enabled:
        output_video_path = PROJECT_ROOT / config.get(
            'paths.annotated_video_file', 'data/videos/output_annotated.mp4'
        )
        annotator = VideoAnnotator(
            str(output_video_path),
            fps=config.get('output.video_fps', 10),
            codec=config.get('output.video_codec', 'mp4v')
        )
        logger.info(f"Video annotation enabled -> {output_video_path}")

    # ---------------- Detection Log ---------------- #
    # Rows buffered in memory so box_id can be backfilled before writing.
    CSV_HEADER = [
        "frame_index", "filename", "timestamp_s",
        "yolo_label", "confidence", "refined_label",
        "track_id", "instance_id", "box_id",
        "entry_detected", "is_new_item", "hand_detected",
        "exit_detected", "exit_verified_by",
    ]
    BOX_ID_COL   = CSV_HEADER.index("box_id")
    INSTANCE_COL = CSV_HEADER.index("instance_id")
    csv_rows: list = []

    class _CsvWriter:
        def writerow(self, row):
            csv_rows.append(list(row))

    csv_writer = _CsvWriter()

    ignoreLabels                = set(config.get('detection.ignore_labels', []))
    detected_boxes_count        = 0
    total_plausibility_discards = 0
    total_exit_events           = 0
    label_cache: dict           = {}
    completed_box_ids: set      = set()   # box IDs fully packed, skip in exit sync
    prev_box_id: str            = 'BOX-001'  # tracks box ID changes across frames
    box_just_changed: bool      = False       # True on the frame a box switch happens
    last_box_frame: object      = None        # last frame image before box ID switched

    logger.info("Running detection pipeline...")

    # ---------------- Main Frame Loop ---------------- #
    for f in framesData:
        originalFrame    = f["frame"]
        frame_idx        = f['index']
        timestamp        = meta_map.get(f.get('filename'), None)
        frame_h, frame_w = originalFrame.shape[:2]

        # ---- Hand detection ----
        hand_detected = hand_detector.detect(originalFrame)
        hand_bboxes   = get_hand_bboxes(hand_detector, originalFrame)

        # ---- YOLO detections ----
        detections = detector.detectObjects(
            originalFrame,
            confThresh=conf_threshold,
            boxConfThresh=box_conf_threshold
        )

        # ---- Separate boxes from items ----
        box_detections_raw  = []
        item_detections_raw = []
        for det in detections:
            if is_box_label(det['label'], box_labels):
                box_detections_raw.append(det)
            elif det['label'].lower() not in ignoreLabels:
                item_detections_raw.append(det)

        # ---- Scan frame for barcodes / QR codes ----
        barcode_scanner.scan(originalFrame, frame_idx)

        # ---- Resolve current box ID ----
        # Always use the most recently scanned barcode as the box ID.
        # If nothing has been scanned yet, fall back to BOX-001.
        FALLBACK_IDS = {'BOX-001', 'box-001'}
        active_codes = barcode_scanner.get_active_codes(frame_idx)
        if active_codes:
            # Pick the code seen most recently this video
            current_box_id = max(
                active_codes.items(),
                key=lambda x: x[1]['last_seen_frame']
            )[0]
        else:
            current_box_id = 'BOX-001'

        box_detections = [{**det, 'box_id': current_box_id} for det in box_detections_raw]

        # ---- Detect box ID change — clear caches so items are re-registered ----
        # When the barcode switches to a new box, the label_cache must be cleared
        # so previously seen track_ids are re-evaluated under the new box context.
        # The entry_detector confirmed_entries are also reset so items that were
        # confirmed in the old box can be confirmed again in the new box.
        box_just_changed = False
        if (current_box_id not in FALLBACK_IDS
                and prev_box_id not in FALLBACK_IDS
                and current_box_id != prev_box_id):
            logger.info(
                f"[BOX-CHANGE] Box switched: '{prev_box_id}' → '{current_box_id}' "
                f"at frame {frame_idx} — clearing label cache and entry state"
            )
            label_cache.clear()
            box_just_changed = True
            if entry_detector:
                entry_detector.confirmed_entries.clear()
                entry_detector.entered_items.clear()
                entry_detector.inside_counter.clear()
                entry_detector.outside_counter.clear()
                entry_detector.track_history.clear()
                entry_detector._last_seen_frame.clear()
            # Clear ALL state from the old box so nothing bleeds into the new box
            logger.info(f"[BOX-CHANGE] Resetting all state for new box '{current_box_id}'")

            # 1. Mark the old box as completed so sync_registry skips its items
            completed_box_ids.add(prev_box_id)
            logger.info(f"[BOX-CHANGE] Marked '{prev_box_id}' as completed")

            # 2. Cancel pending confirmation queue entries from the old box
            if exit_detector:
                for cand in exit_detector._candidates.values():
                    if (cand.confirmation_id
                            and cand.confirmation_id in confirmation_queue
                            and confirmation_queue[cand.confirmation_id]['answer'] is None):
                        confirmation_queue[cand.confirmation_id]['answer'] = False
                        logger.info(
                            f"[BOX-CHANGE] Cancelled exit confirmation for "
                            f"'{cand.label}' from old box '{prev_box_id}'"
                        )
                exit_detector._candidates.clear()
                logger.info("[BOX-CHANGE] Exit candidates cleared")
        # Save this frame as the last known frame for the current box,
        # so if the box switches next frame we have the right image for flush.
        if current_box_id == prev_box_id or prev_box_id in FALLBACK_IDS:
            last_box_frame = originalFrame.copy()
        prev_box_id = current_box_id

        if box_detections:
            detected_boxes_count += len(box_detections)
            logger.info(
                f"Frame {frame_idx}: {len(box_detections)} box(es) | "
                f"{len(item_detections_raw)} item(s) | "
                f"box_id={current_box_id} | hand_detected={hand_detected}"
            )

        # ---- Sync everything to current_box_id ----
        # Update any registry instances, exit candidates, and confirmation
        # queue entries that still carry a fallback ID.
        if current_box_id not in FALLBACK_IDS:
            for inst in registry.get_active_items():
                if inst.box_id in FALLBACK_IDS:
                    inst.box_id = current_box_id
                    logger.info(f"[BARCODE-SYNC] Instance #{inst.instance_id} '{inst.label}' → '{current_box_id}'")
            if exit_detector:
                for cand in exit_detector._candidates.values():
                    if cand.box_id in FALLBACK_IDS:
                        cand.box_id = current_box_id
                        logger.info(f"[BARCODE-SYNC] Exit candidate #{cand.instance_id} '{cand.label}' → '{current_box_id}'")
                    if cand.confirmation_id and cand.confirmation_id in confirmation_queue:
                        if confirmation_queue[cand.confirmation_id].get('box_id') in FALLBACK_IDS:
                            confirmation_queue[cand.confirmation_id]['box_id'] = current_box_id

        # ---- Item tracking ----
        if tracker:
            tracked_objects = tracker.update(item_detections_raw, frame=originalFrame)
        else:
            tracked_objects = [{**det, 'track_id': None} for det in item_detections_raw]

        # ---- Video annotation ----
        if video_output_enabled:
            detections_per_frame[frame_idx] = {
                'detections': tracked_objects + [
                    {**box, 'track_id': None, 'label': box['box_id']}
                    for box in box_detections
                ],
                'hand_detected': hand_detected
            }

        # ---- Notify EntryDetector of all visible tracks ----
        if entry_detector:
            for obj in tracked_objects:
                tid = obj.get('track_id')
                if tid is not None:
                    entry_detector.notify_track_seen(tid, frame_idx)

        # ---- Per-object processing ----
        for obj in tracked_objects:
            track_id   = obj.get('track_id')
            yoloLabel  = obj["label"]
            confidence = obj["confidence"]
            bbox       = obj["bbox"]

            if yoloLabel.lower() in ignoreLabels:
                continue

            furniture_labels = {'bed', 'couch', 'sofa', 'desk', 'dining table'}

            # ---- Fast path: cached track ----
            if track_id is not None and track_id in label_cache:
                refined_label = label_cache[track_id]
                is_uncertain  = False

                entry_detected = False
                box_id         = None

                if entry_detector and box_detections:
                    for box_det in box_detections:
                        box_bbox = box_det['bbox']
                        if item_larger_than_box(bbox, box_bbox):
                            continue
                        confirmed = entry_detector.detect_entry(
                            track_id=track_id,
                            item_bbox=bbox,
                            box_id=box_det['box_id'],
                            box_bbox=box_bbox,
                            frame_number=frame_idx,
                            overlap_threshold=config.get('entry_detection.overlap_threshold', 0.5)
                        )
                        if confirmed:
                            entry_detected = True
                            box_id         = box_det['box_id']
                            logger.info(f"[ENTRY] Frame {frame_idx}: Track#{track_id} '{refined_label}' in {box_id}")
                            break

                if refined_label.lower().strip() in furniture_labels and entry_detected and box_id:
                    csv_writer.writerow([frame_idx, f.get('filename'), timestamp, yoloLabel, confidence, refined_label, track_id, None, None, False, False, hand_detected, False, None])
                    continue

                if entry_enabled and not entry_detected:
                    csv_writer.writerow([frame_idx, f.get('filename'), timestamp, yoloLabel, confidence, None, track_id, None, None, False, False, hand_detected, False, None])
                    continue

            else:
                # ---- Slow path: new track — entry detection then LLaMA ----
                entry_detected = False
                box_id         = None

                if entry_detector and box_detections:
                    for box_det in box_detections:
                        box_bbox = box_det['bbox']
                        if item_larger_than_box(bbox, box_bbox):
                            continue
                        confirmed = entry_detector.detect_entry(
                            track_id=track_id,
                            item_bbox=bbox,
                            box_id=box_det['box_id'],
                            box_bbox=box_bbox,
                            frame_number=frame_idx,
                            overlap_threshold=config.get('entry_detection.overlap_threshold', 0.5)
                        )
                        if confirmed:
                            entry_detected = True
                            box_id         = box_det['box_id']
                            break

                if entry_enabled and not entry_detected:
                    csv_writer.writerow([frame_idx, f.get('filename'), timestamp, yoloLabel, confidence, None, track_id, None, None, False, False, hand_detected, False, None])
                    continue

                # Item is in a box — run LLaMA for label refinement
                x1, y1, x2, y2 = bbox
                crop = originalFrame[y1:y2, x1:x2]

                if crop.size == 0:
                    logger.warning(f"Empty crop for '{yoloLabel}' frame {frame_idx}")
                    refined_label = yoloLabel
                else:
                    box_context_full = registry.get_unique_labels_for_box(box_id or "GLOBAL")
                    uncertain_yolo   = yoloLabel.lower().strip() in UNCERTAIN_YOLO_LABELS
                    box_context      = None if uncertain_yolo else box_context_full

                    if reasoner:
                        refined_label = reasoner.refineDetection(crop, yoloLabel, box_context=box_context)
                    refined_label = refined_label or yoloLabel

                    if reasoner and refined_label.lower().strip() == yoloLabel.lower().strip():
                        original_label = refined_label
                        refined_label  = reasoner.normalize_label(refined_label)
                        if original_label != refined_label:
                            logger.info(f"[NORMALIZE] '{original_label}' → '{refined_label}'")

                    _, discards = plausibility_filter.filter(
                        [{'label': refined_label, 'confidence': confidence, 'bbox': bbox}],
                        frame_w, frame_h
                    )
                    if discards:
                        total_plausibility_discards += 1
                        csv_writer.writerow([frame_idx, f.get('filename'), timestamp, yoloLabel, confidence, refined_label, track_id, None, None, False, False, hand_detected, False, None])
                        continue

                    if refined_label.lower().strip() in furniture_labels and box_id:
                        total_plausibility_discards += 1
                        csv_writer.writerow([frame_idx, f.get('filename'), timestamp, yoloLabel, confidence, refined_label, track_id, None, None, False, False, hand_detected, False, None])
                        continue

                # ---- Save cropped image to yolo_frames ----
                if config.get('output.save_cropped_images', True) and crop is not None and crop.size > 0:
                    safe_label = (refined_label or yoloLabel).replace(' ', '_').replace('/', '_')
                    crop_path  = annotatedDir / f"frame{frame_idx:05d}_track{track_id}_{safe_label}.jpg"
                    cv2.imwrite(str(crop_path), crop)

                if track_id is not None:
                    label_cache[track_id] = refined_label

                is_uncertain = (
                    refined_label.lower().strip() == yoloLabel.lower().strip()
                    and yoloLabel.lower().strip() in UNCERTAIN_YOLO_LABELS
                )

                logger.info(
                    f"[REFINED] Frame {frame_idx}: Track#{track_id} "
                    f"'{yoloLabel}' -> '{refined_label}'"
                )

                if entry_detected:
                    logger.info(f"[ENTRY] Frame {frame_idx}: Track#{track_id} '{refined_label}' in {box_id}")

            # ---- Registry deduplication ----
            instance_id, is_new_item = registry.register_entry(
                track_id=track_id if track_id is not None else -1,
                refined_label=refined_label,
                box_id=box_id or "GLOBAL",
                frame=frame_idx,
                timestamp=timestamp or 0.0,
                yolo_label=yoloLabel,
                is_uncertain=is_uncertain
            )

            if is_new_item:
                logger.info(
                    f"[NEW ITEM] Instance #{instance_id} "
                    f"Track#{track_id} '{yoloLabel}' -> '{refined_label}' "
                    f"-> {box_id} at {timestamp}s"
                )

            csv_writer.writerow([
                frame_idx, f.get('filename'), timestamp,
                yoloLabel, confidence, refined_label,
                track_id, instance_id, box_id,
                entry_detected, is_new_item, hand_detected,
                False, None,
            ])

        # =====================================================================
        # EXIT DETECTION
        # =====================================================================
        # Don't fire exit detection until a real barcode has been confirmed,
        # and never on the frame the box ID just changed — items from the old
        # box disappearing from view should not count as exits in the new box.
        if (exit_detector and entry_detector
                and current_box_id not in FALLBACK_IDS
                and not box_just_changed):
            # Only sync items from the current active box — skip completed boxes
            for inst in registry.get_active_items():
                if inst.box_id in completed_box_ids:
                    continue
                if inst.instance_id not in exit_detector._candidates:
                    from detection.exit_detector import ExitCandidate
                    exit_detector._candidates[inst.instance_id] = ExitCandidate(
                        instance_id = inst.instance_id,
                        label       = inst.refined_label or inst.label,
                        box_id      = inst.box_id,
                        track_ids   = set(inst.track_ids),
                    )
                else:
                    exit_detector._candidates[inst.instance_id].track_ids = set(inst.track_ids)

            exit_detector.update_hand_flags(
                active_track_bboxes = {
                    obj['track_id']: obj['bbox']
                    for obj in tracked_objects
                    if obj.get('track_id') is not None
                },
                hand_bboxes   = hand_bboxes,
                frame_number  = frame_idx,
                hand_detected = hand_detected,
            )

            active_in_box = entry_detector.get_active_track_ids_in_box(
                current_frame   = frame_idx,
                stale_threshold = config.get('exit_detection.stale_threshold', 5),
            )
            box_bboxes_map = {bd['box_id']: bd['bbox'] for bd in box_detections}

            prev_pending = {
                cid for cid, e in confirmation_queue.items() if e['answer'] is None
            }

            confirmed_removals = exit_detector.update_absences(
                active_track_ids_in_box = active_in_box,
                box_bboxes              = box_bboxes_map,
                frame                   = originalFrame,
                frame_number            = frame_idx,
                timestamp               = timestamp or 0.0,
                registry                = registry,
                hand_detected           = hand_detected,
            )

            new_pending = {
                cid for cid, e in confirmation_queue.items() if e['answer'] is None
            }
            if new_pending - prev_pending:
                play_alert()
                logger.info("[EXIT] 🔔 Audio alert played — awaiting user confirmation")

            # ---- Re-sync confirmation queue after exit detection ----
            # Patch any queue entries posted this frame that still carry BOX-001.
            if current_box_id not in FALLBACK_IDS:
                for cid, entry in confirmation_queue.items():
                    if entry.get('box_id') in FALLBACK_IDS:
                        entry['box_id'] = current_box_id

            for instance_id, label, box_id in confirmed_removals:
                registry.mark_removed(instance_id, frame_idx, timestamp or 0.0)
                total_exit_events += 1
                logger.info(
                    f"[EXIT] ✓ '{label}' (#{instance_id}) removed from "
                    f"{box_id} at t={timestamp:.2f}s [verified_by=human]"
                )
                csv_writer.writerow([
                    frame_idx, f.get('filename'), timestamp,
                    "EXIT_EVENT", None, label,
                    None, instance_id, box_id,
                    False, False, hand_detected,
                    True, "human",
                ])

    # =========================================================================
    # END-OF-VIDEO: wait for pending human confirmations
    # =========================================================================
    if exit_detector:
        logger.info("Video processing complete. Checking end-of-video exit confirmations...")

        if entry_detector and framesData:
            entry_detector.get_active_track_ids_in_box(
                current_frame   = framesData[-1]['index'],
                stale_threshold = config.get('exit_detection.stale_threshold', 5),
            )
            exit_detector.sync_registry(registry)

        last_frame     = framesData[-1]['index'] if framesData else 0
        last_ts        = meta_map.get(framesData[-1].get('filename'), 0.0) if framesData else 0.0
        last_frame_img = framesData[-1]['frame'] if framesData else None

        for cand in exit_detector.get_candidates().values():
            if cand.confirmed_removed or cand.user_queried:
                continue

            full_video_frames = framesData[-1]['index'] if framesData else 0
            very_long_absence = cand.absent_frames >= int(full_video_frames * 0.6)

            if cand.absent_frames == 0 and not cand.ever_hand_flagged:
                continue

            if not cand.ever_hand_flagged and not very_long_absence:
                logger.debug(
                    f"[EXIT-FLUSH] Skipping #{cand.instance_id} '{cand.label}' — "
                    f"no hand interaction, likely YOLO dropout"
                )
                continue

            logger.info(
                f"[EXIT-FLUSH] Requesting confirmation for "
                f"#{cand.instance_id} '{cand.label}' "
                f"(absent={cand.absent_frames}, ever_hand_flagged={cand.ever_hand_flagged})"
            )
            img = last_frame_img if last_frame_img is not None else np.zeros((100, 100, 3), dtype=np.uint8)
            conf_id = exit_detector._verifier.request_confirmation(
                box_crop     = img,
                item_label   = cand.label,
                box_id       = cand.box_id,
                instance_id  = cand.instance_id,
                frame_number = last_frame,
                timestamp    = last_ts,
            )
            cand.user_queried    = True
            cand.confirmation_id = conf_id

        pending_ids = {
            cand.confirmation_id
            for cand in exit_detector.get_candidates().values()
            if cand.user_queried and not cand.confirmed_removed and cand.confirmation_id
            and confirmation_queue.get(cand.confirmation_id, {}).get('answer') is None
        }

        if pending_ids:
            play_alert()
            logger.info(f"[EXIT-FLUSH] 🔔 Waiting for {len(pending_ids)} end-of-video confirmation(s)...")
            while True:
                if all(
                    confirmation_queue.get(cid, {}).get('answer') is not None
                    for cid in pending_ids
                ):
                    break
                time.sleep(1.0)
            logger.info("[EXIT-FLUSH] All confirmations received.")

        for cand in exit_detector.get_candidates().values():
            if cand.confirmed_removed or not cand.user_queried or not cand.confirmation_id:
                continue
            answer = confirmation_queue.get(cand.confirmation_id, {}).get('answer')
            if answer is True:
                registry.mark_removed(cand.instance_id, last_frame, last_ts or 0.0)
                cand.confirmed_removed = True
                total_exit_events += 1
                logger.info(f"[EXIT-FLUSH] ✓ '{cand.label}' (#{cand.instance_id}) removed [verified_by=human]")
                csv_writer.writerow([
                    last_frame,
                    framesData[-1].get('filename') if framesData else None,
                    last_ts, "EXIT_EVENT", None, cand.label,
                    None, cand.instance_id, cand.box_id,
                    False, False, False, True, "human",
                ])
            else:
                logger.info(f"[EXIT-FLUSH] ✗ '{cand.label}' (#{cand.instance_id}) rejected by user — keeping as present")

    # ---------------- Annotated video ---------------- #
    if annotator:
        logger.info("Generating annotated video...")
        for fd in framesData:
            fi         = fd['index']
            ts         = meta_map.get(fd.get('filename'))
            frame_data = detections_per_frame.get(fi, {})
            annotator.add_frame(
                fd['frame'],
                frame_data.get('detections', []),
                fi,
                ts,
                frame_data.get('hand_detected', False)
            )
        annotator.finalize()

    # ---------------- Export results ---------------- #
    unique_labels = registry.get_unique_labels()
    all_items     = registry.get_all_items()

    logger.info(f"Unique items detected   : {len(unique_labels)}")
    logger.info(f"Total instances         : {len(all_items)}")
    logger.info(f"Total box detections    : {detected_boxes_count}")
    logger.info(f"Plausibility discards   : {total_plausibility_discards}")
    logger.info(f"Exit events confirmed   : {total_exit_events}")

    # Write refined item list grouped by box
    with open(final_output_file, 'w', encoding='utf-8') as fout:
        all_box_ids = sorted(set(
            inst.box_id for inst in all_items
            if inst.box_id not in ('GLOBAL', '', None)
        ))
        if all_box_ids:
            for box_id in all_box_ids:
                box_labels_list = registry.get_unique_labels_for_box(box_id)
                # Also include removed items for this box
                removed_labels = sorted(set(
                    inst.refined_label or inst.label
                    for inst in all_items
                    if inst.box_id == box_id and inst.refined_label
                ))
                combined = sorted(set(box_labels_list) | set(removed_labels))
                fout.write(f"[{box_id}]\n")
                for idx, name in enumerate(combined, 1):
                    fout.write(f'  item {idx}: "{name.title()}"\n')
                fout.write("\n")
        else:
            for idx, name in enumerate(unique_labels, 1):
                fout.write(f'item {idx}: "{name.title()}"\n')
    logger.info(f"Item list saved -> {final_output_file}")

    registry_file = PROJECT_ROOT / 'item_registry.json'
    with open(registry_file, 'w', encoding='utf-8') as fout:
        json.dump(registry.export_to_dict(), fout, indent=2)
    logger.info(f"Registry saved -> {registry_file}")

    entry_log_file = PROJECT_ROOT / 'entry_log.json'
    with open(entry_log_file, 'w', encoding='utf-8') as fout:
        json.dump({
            'total_entries': len(all_items),
            'entries': [
                {
                    'instance_id': inst.instance_id,
                    'track_id':    sorted(list(inst.track_ids))[0],
                    'label':       inst.refined_label or inst.label,
                    'box_id':      inst.box_id,
                    'timestamp':   inst.first_ts,
                    'frame':       inst.first_frame,
                    'status':      inst.status,
                    'exit_frame':  inst.exit_frame,
                    'exit_ts':     inst.exit_ts,
                }
                for inst in all_items
            ]
        }, fout, indent=2)
    logger.info(f"Entry log saved -> {entry_log_file}")

    export = registry.export_to_dict()
    box_mapping_file = PROJECT_ROOT / 'box_mappings.json'
    with open(box_mapping_file, 'w', encoding='utf-8') as fout:
        json.dump({
            'boxes': {
                box_id: [
                    {
                        'track_id':  sorted(list(inst['track_ids']))[0],
                        'label':     inst['label'],
                        'timestamp': inst['first_ts'],
                        'status':    inst['status'],
                    }
                    for inst in items
                ]
                for box_id, items in export['by_box'].items()
            },
            'summary': {
                'total_boxes':   len(export['by_box']),
                'total_items':   export['summary']['total_unique_items'],
                'items_in_box':  export['summary']['items_in_box'],
                'items_removed': export['summary']['items_removed'],
            }
        }, fout, indent=2)
    logger.info(f"Box mappings saved -> {box_mapping_file}")

    # barcode_scan_log.json — all unique codes detected during this run
    barcode_log_file = PROJECT_ROOT / 'barcode_scan_log.json'
    with open(barcode_log_file, 'w', encoding='utf-8') as fout:
        json.dump(barcode_scanner.summary(), fout, indent=2)
    logger.info(f"Barcode scan log saved -> {barcode_log_file}")

    if box_tracker:
        active_boxes = box_tracker.get_active_tracks()
        logger.info(
            f"BoxTracker: {len(active_boxes)} active track(s) at pipeline end → "
            + (", ".join(t['box_id'] for t in active_boxes) if active_boxes else "none")
        )

    # ---------------- Backfill detection log box IDs ---------------- #
    all_scanned  = barcode_scanner.summary().get('codes', {})
    FALLBACK_IDS = {'BOX-001', 'box-001', 'GLOBAL', '', 'None', 'nan'}

    best_barcode_id = None
    if all_scanned:
        sorted_codes = sorted(
            [(d, i) for d, i in all_scanned.items() if d not in FALLBACK_IDS],
            key=lambda x: x[1]['first_frame']
        )
        if sorted_codes:
            best_barcode_id = sorted_codes[0][0]

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
        elif best_barcode_id and str(row[BOX_ID_COL]) in FALLBACK_IDS:
            row[BOX_ID_COL] = best_barcode_id

    with open(detection_log_file, 'w', newline='', encoding='utf-8') as _csvf:
        _w = csv.writer(_csvf)
        _w.writerow(CSV_HEADER)
        _w.writerows(csv_rows)
    logger.info(f"Detection log saved (backfilled) -> {detection_log_file}")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Total processing time: {elapsed:.2f}s")
    logger.info("="*60)
    logger.info("Pipeline Complete!")
    logger.info("="*60)


# ---------------- Runner ---------------- #
if __name__ == "__main__":
    videoPath = sys.argv[1] if len(sys.argv) > 1 else None
    main(videoPath)