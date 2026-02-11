#############################################################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'main.py'
#
# Description:
# Main entry point for the Vision-Based-Packing-Project pipeline.
# Extracts frames, runs YOLO detections, then refines labels using LLaMA
# vision reasoning through Ollama. Entry detection works for items already
# in boxes OR items entering boxes.
#
# Item deduplication is handled by ItemRegistry (stable instance IDs).
# Box identity is handled by BoxTracker (stable box IDs across frames).
###############################################################################

from video_processor                import extractFrames
from detection.frame_loader         import loadFrames
from detection.object_detector      import ObjectDetector
from detection.frame_reasoner       import FrameReasoner
from detection.object_tracker       import ObjectTracker
from detection.video_annotator      import VideoAnnotator
from detection.qr_detector          import QRDetector, BoxItemMapper
from detection.entry_detector       import EntryDetector
from detection.plausibility_filter  import PlausibilityFilter
from detection.item_registry        import ItemRegistry
import sys, os
sys.path.insert(0, '..')
from config_loader                  import get_config
import cv2
from pathlib import Path
import csv
import json
import logging
from datetime import datetime

# ------------------ Project Root ------------------ #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------ Logging Setup ----------------- #
def setup_logging(config):
    if sys.platform == 'win32':
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

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


# ------------------ Main Pipeline ----------------- #
def main(videoPath=None):
    start_time = datetime.now()

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
    yolo_model         = PROJECT_ROOT / config.get('detection.yolo_model',  'models/yolo26m.pt')
    box_model_path     = PROJECT_ROOT / config.get('detection.box_model',   'models/cardboard_boxYOLO.pt')
    conf_threshold     = config.get('detection.confidence_threshold',       0.35)
    box_conf_threshold = config.get('detection.box_confidence_threshold',   conf_threshold)
    box_labels         = config.get('detection.box_labels', ['box', 'carton', 'cardboard'])

    logger.info(f"Initializing YOLO models: {yolo_model}, {box_model_path}")
    logger.info(f"Item conf: {conf_threshold} | Box conf: {box_conf_threshold}")
    detector = ObjectDetector(str(yolo_model), str(box_model_path))

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

    # ---------------- Item Registry ---------------- #
    same_item_window = config.get('item_registry.same_item_window', 60)
    registry = ItemRegistry(same_item_window=same_item_window)
    logger.info(f"ItemRegistry initialised (same_item_window={same_item_window} frames)")

    # ---------------- Entry Detector ---------------- #
    entry_enabled  = config.get('entry_detection.enabled', True)
    entry_detector = None
    if entry_enabled:
        entry_detector = EntryDetector(
            entry_threshold=config.get('entry_detection.entry_threshold', 3),
            exit_threshold=config.get('entry_detection.exit_threshold',   5),
            require_motion=config.get('entry_detection.require_motion',   False),
        )
        logger.info(
            f"Entry detection enabled | "
            f"threshold={config.get('entry_detection.entry_threshold', 3)} | "
            f"overlap={config.get('entry_detection.overlap_threshold', 0.5)} | "
            f"require_motion={config.get('entry_detection.require_motion', False)}"
        )

    # ---------------- LLaMA Reasoner ---------------- #
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
    csv_file   = open(detection_log_file, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_index", "filename", "timestamp_s",
        "yolo_label", "confidence", "refined_label",
        "track_id", "instance_id", "box_id",
        "entry_detected", "is_new_item", "plausibility_discarded"
    ])

    ignoreLabels                = set(config.get('detection.ignore_labels', []))
    detected_boxes_count        = 0
    total_plausibility_discards = 0

    logger.info("Running detection pipeline...")

    # ---------------- Main Frame Loop ---------------- #
    for f in framesData:
        originalFrame    = f["frame"]
        frame_idx        = f['index']
        timestamp        = meta_map.get(f.get('filename'), None)
        frame_h, frame_w = originalFrame.shape[:2]

        processed_instances = set()

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

        # ---- Assign placeholder box ID ----
        # All detected boxes get the same ID for now.
        # Will be replaced with barcode-scanned IDs later.
        box_detections = [{**det, 'box_id': 'BOX-001'} for det in box_detections_raw]

        if box_detections:
            detected_boxes_count += len(box_detections)
            logger.debug(
                f"Frame {frame_idx}: boxes active = "
                f"{[b['box_id'] for b in box_detections]}"
            )

        # All item detections pass through to LLaMA.
        # Plausibility check runs AFTER refinement so LLaMA gets a chance
        # to correct mislabels (e.g. YOLO: "refrigerator" -> LLaMA: "Tissue Box").
        item_detections = item_detections_raw

        if box_detections:
            logger.info(
                f"Frame {frame_idx}: {len(box_detections)} box(es) | "
                f"{len(item_detections)} item(s)"
            )

        # ---- Item tracking ----
        if tracker:
            tracked_objects = tracker.update(item_detections, frame=originalFrame)
        else:
            tracked_objects = [{**det, 'track_id': None} for det in item_detections]

        # ---- Video annotation ----
        if video_output_enabled:
            detections_per_frame[frame_idx] = tracked_objects + [
                {**box, 'track_id': None, 'label': box['box_id']}
                for box in box_detections
            ]

        # ---- Per-object processing ----
        for obj in tracked_objects:
            track_id   = obj.get('track_id')
            yoloLabel  = obj["label"]
            confidence = obj["confidence"]
            bbox       = obj["bbox"]

            if yoloLabel.lower() in ignoreLabels:
                continue

            # ---- Entry detection ----
            entry_detected = False
            box_id         = None

            if entry_detector and box_detections:
                for box_det in box_detections:
                    box_bbox = box_det['bbox']

                    if item_larger_than_box(bbox, box_bbox):
                        logger.debug(
                            f"Frame {frame_idx}: Skipping Track#{track_id} '{yoloLabel}' "
                            f"- item area ({get_bbox_area(bbox)}px2) > "
                            f"box area ({get_bbox_area(box_bbox)}px2)"
                        )
                        continue

                    confirmed = entry_detector.detect_entry(
                        track_id=track_id,
                        item_bbox=bbox,
                        box_id=box_det['box_id'],       # stable ID from BoxTracker
                        box_bbox=box_bbox,
                        frame_number=frame_idx,
                        overlap_threshold=config.get('entry_detection.overlap_threshold', 0.5)
                    )
                    if confirmed:
                        entry_detected = True
                        box_id         = box_det['box_id']
                        logger.info(
                            f"[ENTRY] Frame {frame_idx}: "
                            f"Track#{track_id} '{yoloLabel}' in {box_id}"
                        )
                        break

            # ---- Skip if not in a box ----
            if entry_enabled and not entry_detected:
                csv_writer.writerow([
                    frame_idx, f.get('filename'), timestamp,
                    yoloLabel, confidence,
                    None, track_id, None, None,
                    False, False, False
                ])
                continue

            # ---- Registry deduplication ----
            instance_id, is_new_item = registry.register_entry(
                track_id=track_id if track_id is not None else -1,
                label=yoloLabel,
                box_id=box_id or "GLOBAL",
                frame=frame_idx,
                timestamp=timestamp or 0.0
            )

            # ---- LLaMA refinement (new items only) ----
            refined_label = None

            if is_new_item and instance_id not in processed_instances:
                x1, y1, x2, y2 = bbox
                crop = originalFrame[y1:y2, x1:x2]

                if crop.size == 0:
                    logger.warning(f"Empty crop for '{yoloLabel}' frame {frame_idx}")
                    refined_label = yoloLabel
                else:
                    if config.get('output.save_cropped_images', True):
                        cv2.imwrite(
                            str(annotatedDir /
                                f"frame{frame_idx:05d}_inst{instance_id}_{yoloLabel}.jpg"),
                            crop
                        )

                    if reasoner:
                        refined_label = reasoner.refineDetection(crop, yoloLabel)
                    refined_label = refined_label or yoloLabel

                    # ---- Plausibility check on REFINED label ----
                    # If LLaMA corrected "refrigerator" -> "Tissue Box" it passes.
                    # If LLaMA confirmed "refrigerator" or fell back to it, discard.
                    _, discards = plausibility_filter.filter(
                        [{'label': refined_label, 'confidence': confidence, 'bbox': bbox}],
                        frame_w, frame_h
                    )
                    if discards:
                        total_plausibility_discards += 1
                        logger.info(
                            f"[PLAUSIBILITY] Instance #{instance_id} discarded after LLaMA: "
                            f"YOLO='{yoloLabel}' -> LLaMA='{refined_label}' "
                            f"(bbox too small for '{refined_label}')"
                        )
                        # Remove from registry — it turned out to be implausible
                        registry._instances.pop(instance_id, None)
                        registry._track_to_instance.pop(
                            track_id if track_id is not None else -1, None
                        )
                        csv_writer.writerow([
                            frame_idx, f.get('filename'), timestamp,
                            yoloLabel, confidence, refined_label,
                            track_id, instance_id, box_id,
                            entry_detected, is_new_item, True
                        ])
                        continue

                    registry.set_refined_label(instance_id, refined_label)
                    processed_instances.add(instance_id)

                    logger.info(
                        f"[NEW ITEM] Instance #{instance_id} "
                        f"Track#{track_id} '{yoloLabel}' -> '{refined_label}' "
                        f"-> {box_id} at {timestamp}s"
                    )
            else:
                inst          = registry.get_instance(instance_id)
                refined_label = (inst.refined_label or inst.label) if inst else yoloLabel
                if not is_new_item:
                    logger.debug(
                        f"[RE-DET] Instance #{instance_id} Track#{track_id} "
                        f"'{yoloLabel}' (already recorded as '{refined_label}')"
                    )

            # ---- CSV row ----
            csv_writer.writerow([
                frame_idx, f.get('filename'), timestamp,
                yoloLabel, confidence, refined_label,
                track_id, instance_id, box_id,
                entry_detected, is_new_item, False
            ])

    # ---------------- Annotated video ---------------- #
    if annotator:
        logger.info("Generating annotated video...")
        for fd in framesData:
            fi = fd['index']
            ts = meta_map.get(fd.get('filename'))
            annotator.add_frame(
                fd['frame'], detections_per_frame.get(fi, []), fi, ts
            )
        annotator.finalize()

    # ---------------- Export results ---------------- #
    unique_labels = registry.get_unique_labels()
    all_items     = registry.get_all_items()

    logger.info(f"Unique items detected   : {len(unique_labels)}")
    logger.info(f"Total instances         : {len(all_items)}")
    logger.info(f"Total box detections    : {detected_boxes_count}")
    logger.info(f"Plausibility discards   : {total_plausibility_discards}")
    # refined_item_list.txt
    with open(final_output_file, 'w', encoding='utf-8') as fout:
        for idx, name in enumerate(unique_labels, 1):
            fout.write(f'item {idx}: "{name.title()}"\n')
    logger.info(f"Item list saved -> {final_output_file}")

    # item_registry.json
    registry_file = PROJECT_ROOT / 'item_registry.json'
    with open(registry_file, 'w', encoding='utf-8') as fout:
        json.dump(registry.export_to_dict(), fout, indent=2)
    logger.info(f"Registry saved -> {registry_file}")

    # entry_log.json (backwards compat)
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
                }
                for inst in all_items
            ]
        }, fout, indent=2)

    # box_mappings.json (backwards compat)
    export = registry.export_to_dict()
    box_mapping_file = PROJECT_ROOT / 'box_mappings.json'
    with open(box_mapping_file, 'w', encoding='utf-8') as fout:
        json.dump({
            'boxes': {
                box_id: [
                    {
                        'track_id':  sorted(list(inst['track_ids']))[0],
                        'label':     inst['label'],
                        'timestamp': inst['first_ts']
                    }
                    for inst in items
                ]
                for box_id, items in export['by_box'].items()
            },
            'summary': {
                'total_boxes': len(export['by_box']),
                'total_items': export['summary']['total_unique_items'],
            }
        }, fout, indent=2)

    csv_file.close()
    logger.info(f"Detection log saved -> {detection_log_file}")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Total processing time: {elapsed:.2f}s")
    logger.info("="*60)
    logger.info("Pipeline Complete!")
    logger.info("="*60)


# ---------------- Runner ---------------- #
if __name__ == "__main__":
    videoPath = sys.argv[1] if len(sys.argv) > 1 else None
    main(videoPath)