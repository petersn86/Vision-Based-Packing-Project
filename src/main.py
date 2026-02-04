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
###############################################################################

from video_processor            import extractFrames
from detection.frame_loader     import loadFrames
from detection.object_detector  import ObjectDetector
from detection.frame_reasoner   import FrameReasoner
from detection.object_tracker   import ObjectTracker
from detection.video_annotator  import VideoAnnotator
from detection.qr_detector      import QRDetector, BoxItemMapper
from detection.entry_detector   import EntryDetector
import sys, os
sys.path.insert(0, '..')        # Add parent directory to path
from config_loader              import get_config
import cv2
from pathlib import Path
import csv
import json
import logging
from datetime import datetime

# ------------------ Project Root ------------------ #
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # One level above src

# ------------------ Logging Setup ----------------- #
def setup_logging(config):
    """Configure logging based on config settings with Windows Unicode support"""
    if sys.platform == 'win32':
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
    
    log_level = getattr(logging, config.get('logging.level', 'INFO'))
    log_file = config.get('logging.log_file', 'app.log')
    if log_file:
        log_file = PROJECT_ROOT / log_file  # Save log at project root
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

# ------------------ Main Pipeline ----------------- #
def main(videoPath=None):
    start_time = datetime.now()
    
    # Load configuration
    config = get_config()
    logger = setup_logging(config)
    
    logger.info("="*60)
    logger.info("Vision-Based Packing Project - YOLO Box Entry Detection")
    logger.info("="*60)

    # ---------------- Paths ---------------- #
    framesDir = PROJECT_ROOT / config.get('paths.frames_dir', 'data/frames')
    framesDir.mkdir(parents=True, exist_ok=True)

    annotatedDir = PROJECT_ROOT / config.get('paths.annotated_dir', 'data/yolo_frames')
    annotatedDir.mkdir(parents=True, exist_ok=True)

    final_output_file = PROJECT_ROOT / config.get('paths.refined_items_file', 'refined_item_list.txt')
    detection_log_file = PROJECT_ROOT / config.get('paths.detection_log_file', 'detection_log.csv')

    if videoPath is None:
        videoPath = PROJECT_ROOT / config.get('video.default_video', 'data/videos/packing_video.mp4')

    # ---------------- Frame Extraction ---------------- #
    logger.info(f"Extracting frames from video: {videoPath}")
    frame_interval = config.get('video.frame_interval', 2.0)
    frames_meta = extractFrames(str(videoPath), str(framesDir), frame_interval)

    framesData = loadFrames(str(framesDir))
    logger.info(f"Loaded {len(framesData)} frames for processing")

    # Build mapping from filename -> timestamp
    meta_map = {}
    try:
        meta_map = {os.path.basename(m.get('path', '')): m.get('timestamp', None) for m in frames_meta}
    except Exception as e:
        logger.warning(f"Could not build timestamp mapping: {e}")

    # ---------------- Object Detector ---------------- #
    yolo_model = PROJECT_ROOT / config.get('detection.yolo_model', 'models/yolo11l.pt')
    box_model = PROJECT_ROOT / config.get('detection.box_model', 'models/cardboard_boxYOLO.pt')
    conf_threshold = config.get('detection.confidence_threshold', 0.35)
    box_conf_threshold = config.get('detection.box_confidence_threshold', conf_threshold)
    
    logger.info(f"Initializing YOLO models: {yolo_model}, {box_model}")
    logger.info(f"Item confidence threshold: {conf_threshold}")
    logger.info(f"Box confidence threshold: {box_conf_threshold}")
    detector = ObjectDetector(str(yolo_model), str(box_model))

    # ---------------- Tracker ---------------- #
    tracking_enabled = config.get('tracking.enabled', True)
    tracker = None
    if tracking_enabled:
        track_thresh = config.get('tracking.track_thresh')
        track_buffer = config.get('tracking.track_buffer')
        match_thresh = config.get('tracking.match_thresh')
        second_match_thresh = config.get('tracking.second_match_thresh')
        
        if track_thresh is not None:
            logger.info(f"Object tracking enabled (ByteTrack mode)")
            tracker = ObjectTracker(
                track_thresh=track_thresh or 0.5,
                track_buffer=track_buffer or 30,
                match_thresh=match_thresh or 0.8,
                second_match_thresh=second_match_thresh or 0.5
            )
        else:
            max_age = config.get('tracking.max_age', 30)
            logger.info(f"Object tracking enabled (SORT compatibility mode)")
            tracker = ObjectTracker(
                track_thresh=0.5,
                track_buffer=max_age,
                match_thresh=0.8,
                second_match_thresh=0.5
            )
    else:
        logger.info("Object tracking disabled")

    # ---------------- Box Detection (YOLO-based, no QR codes) ---------------- #
    box_mapper = BoxItemMapper()
    logger.info("Box detection via YOLO (no QR codes required)")

    # ---------------- Entry Detection (YOLO boxes) ---------------- #
    entry_enabled = config.get('entry_detection.enabled', True)
    entry_detector = None
    if entry_enabled:
        entry_threshold = config.get('entry_detection.entry_threshold', 3)
        exit_threshold = config.get('entry_detection.exit_threshold', 5)
        overlap_threshold = config.get('entry_detection.overlap_threshold', 0.5)
        require_motion = config.get('entry_detection.require_motion', False)
        
        logger.info(f"Entry detection enabled (using YOLO-detected boxes)")
        logger.info(f"  entry_threshold={entry_threshold}, overlap={overlap_threshold}")
        logger.info(f"  require_motion={require_motion} (items must move into boxes: {require_motion})")
        
        entry_detector = EntryDetector(entry_threshold, exit_threshold, require_motion)
    else:
        logger.info("Entry detection disabled - tracking ALL detections")

    # ---------------- LLaMA Reasoner ---------------- #
    llama_enabled = config.get('llama.enabled', True)
    reasoner = None
    if llama_enabled:
        llama_model = config.get('llama.model_name', 'llama3.2-vision')
        logger.info(f"Initializing LLaMA reasoner with model: {llama_model}")
        reasoner = FrameReasoner(model_name=llama_model)
    else:
        logger.info("LLaMA refinement disabled")

    # ---------------- Video Annotator ---------------- #
    video_output_enabled = config.get('output.create_video', True)
    annotator = None
    detections_per_frame = {}
    
    if video_output_enabled:
        output_video_path = PROJECT_ROOT / config.get('paths.annotated_video_file', 'data/videos/output_annotated.mp4')
        video_fps = config.get('output.video_fps', 10)
        video_codec = config.get('output.video_codec', 'mp4v')
        logger.info(f"Video annotation enabled (output: {output_video_path})")
        annotator = VideoAnnotator(str(output_video_path), fps=video_fps, codec=video_codec)

    logger.info("Running detection pipeline...")

    # ---------------- Detection Log ---------------- #
    csv_file = open(detection_log_file, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_index", "filename", "timestamp_s", "yolo_label", "confidence", 
                         "refined_label", "track_id", "box_id", "entry_detected"])

    finalItems = set()
    ignoreLabels = set(config.get('detection.ignore_labels', []))
    processed_tracks = set()
    items_entered_boxes = {}  # track_id -> {label, box_id, timestamp}
    detected_boxes_count = 0

    # ---------------- Main Frame Loop ---------------- #
    for f in framesData:
        originalFrame = f["frame"]
        frame_idx = f['index']
        timestamp = meta_map.get(f.get('filename'), None)

        # Get all detections
        detections = detector.detectObjects(originalFrame, confThresh=conf_threshold, boxConfThresh=box_conf_threshold)

        # Separate box detections from item detections
        box_detections = []
        item_detections = []
        
        for det in detections:
            label_lower = det['label'].lower()
            # Check if it's a box/cardboard detection
            if 'box' in label_lower or 'cardboard' in label_lower:
                box_detections.append(det)
                logger.debug(f"Frame {frame_idx}: Detected box at {det['bbox']}")
            elif label_lower not in ignoreLabels:
                item_detections.append(det)
        
        if box_detections:
            detected_boxes_count += len(box_detections)
            logger.info(f"Frame {frame_idx}: Found {len(box_detections)} box(es), {len(item_detections)} item(s)")

        # Object Tracking (only track items, not boxes)
        tracked_objects = []
        if tracker:
            tracked_objects = tracker.update(item_detections)
            logger.debug(f"Frame {frame_idx}: {len(tracked_objects)} tracked items, {len(box_detections)} boxes")
        else:
            tracked_objects = [{**det, 'track_id': None} for det in item_detections]

        # For video annotation, include both items and boxes
        if video_output_enabled:
            all_detections = tracked_objects + [{**box, 'track_id': None, 'label': f'BOX-{i}'} for i, box in enumerate(box_detections)]
            detections_per_frame[frame_idx] = all_detections

        # Process each tracked item
        for obj in tracked_objects:
            track_id = obj.get('track_id')
            yoloLabel = obj["label"]
            confidence = obj["confidence"]
            bbox = obj["bbox"]

            if yoloLabel.lower() in ignoreLabels:
                continue

            # Entry Detection Logic (using YOLO-detected boxes)
            entry_detected = False
            box_id = None
            
            if entry_detector and box_detections:
                # Check entry for each detected box in this frame
                for box_idx, box_det in enumerate(box_detections):
                    box_bbox = box_det['bbox']
                    box_id_temp = f"BOX-{frame_idx}-{box_idx}"
                    
                    confirmed_entry = entry_detector.detect_entry(
                        track_id=track_id,
                        item_bbox=bbox,
                        box_id=box_id_temp,
                        box_bbox=box_bbox,
                        frame_number=frame_idx,
                        overlap_threshold=config.get('entry_detection.overlap_threshold', 0.5)
                    )
                    
                    if confirmed_entry:
                        entry_detected = True
                        box_id = box_id_temp
                        logger.info(f"🎯 [ENTRY DETECTED] Frame {frame_idx}: Track#{track_id} '{yoloLabel}' in {box_id}")
                        break
            
            # Only process items that have entered boxes (or if entry detection is disabled)
            should_process = not entry_enabled or (entry_detected and track_id not in items_entered_boxes)
            
            if should_process:
                
                # Skip if already processed
                if tracker and track_id is not None and track_id in processed_tracks:
                    continue

                x1, y1, x2, y2 = bbox
                croppedImage = originalFrame[y1:y2, x1:x2]
                if croppedImage.size == 0:
                    logger.warning(f"Skipping empty crop for '{yoloLabel}' in frame {frame_idx}")
                    continue

                # Save cropped image if enabled
                if config.get('output.save_cropped_images', True):
                    crop_filename = annotatedDir / f"frame{frame_idx:05d}_track{track_id}_{yoloLabel}.jpg"
                    cv2.imwrite(str(crop_filename), croppedImage)

                # Refine with LLaMA
                refinedLabel = None
                if reasoner:
                    refinedLabel = reasoner.refineDetection(croppedImage, yoloLabel)
                    if track_id is not None:
                        processed_tracks.add(track_id)
                else:
                    refinedLabel = yoloLabel
                
                # Fallback to YOLO if LLaMA returns None
                if refinedLabel is None:
                    refinedLabel = yoloLabel
                    logger.debug(f"Using YOLO label '{yoloLabel}' as fallback")

                # Record the entry
                if refinedLabel and entry_detected:
                    items_entered_boxes[track_id] = {
                        'label': refinedLabel,
                        'box_id': box_id,
                        'timestamp': timestamp,
                        'frame': frame_idx
                    }
                    
                    if box_mapper and box_id:
                        box_mapper.add_mapping(track_id or -1, refinedLabel, box_id, timestamp or 0.0)
                        logger.info(f"📦 [PACKED] '{refinedLabel}' → {box_id} at {timestamp}s")

                # Log results
                if refinedLabel:
                    if refinedLabel not in finalItems:
                        logger.info(f"[NEW ITEM] {timestamp}s Track#{track_id} '{yoloLabel}' → '{refinedLabel}'")
                        finalItems.add(refinedLabel)
                    else:
                        logger.debug(f"[CONFIRMED] {timestamp}s Track#{track_id} '{refinedLabel}'")

                csv_writer.writerow([frame_idx, f.get('filename'), timestamp, yoloLabel, 
                                   confidence, refinedLabel, track_id, box_id, entry_detected])
            else:
                # Item tracked but hasn't entered a box yet
                csv_writer.writerow([frame_idx, f.get('filename'), timestamp, yoloLabel, 
                                   confidence, None, track_id, None, False])

    # ---------------- Generate Annotated Video ---------------- #
    if annotator:
        logger.info("Generating annotated video...")
        for frame_data in framesData:
            frame_idx = frame_data['index']
            frame = frame_data['frame']
            timestamp = meta_map.get(frame_data.get('filename'))
            detections = detections_per_frame.get(frame_idx, [])
            annotator.add_frame(frame, detections, frame_idx, timestamp)
        annotator.finalize()

    logger.info("Processing Complete")

    # ---------------- Export Results ---------------- #
    finalItemsList = sorted(list(finalItems))
    logger.info(f"Found {len(finalItemsList)} unique refined items")
    logger.info(f"Total box detections across all frames: {detected_boxes_count}")
    
    # NEW FORMAT: item: "description"
    with open(final_output_file, 'w', encoding='utf-8') as f:
        for idx, name in enumerate(finalItemsList, 1):
            formatted_name = name.title()
            f.write(f'item {idx}: "{formatted_name}"\n')
    
    logger.info(f"Items saved to {final_output_file}")

    # Export box mappings
    if box_mapper:
        box_mappings = box_mapper.export_to_dict()
        box_mapping_file = PROJECT_ROOT / 'box_mappings.json'
        with open(box_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(box_mappings, f, indent=2)
        logger.info(f"Box mappings saved to {box_mapping_file}")
        logger.info(f"Total boxes with items: {box_mappings['summary']['total_boxes']}")
        logger.info(f"Total items mapped: {box_mappings['summary']['total_items']}")

    # Export entry details
    if entry_detector and items_entered_boxes:
        entry_log_file = PROJECT_ROOT / 'entry_log.json'
        entry_data = {
            'total_entries': len(items_entered_boxes),
            'entries': [
                {
                    'track_id': tid,
                    'label': info['label'],
                    'box_id': info['box_id'],
                    'timestamp': info['timestamp'],
                    'frame': info['frame']
                }
                for tid, info in items_entered_boxes.items()
            ]
        }
        with open(entry_log_file, 'w', encoding='utf-8') as f:
            json.dump(entry_data, f, indent=2)
        logger.info(f"Entry log saved to {entry_log_file}")
        logger.info(f"Total items that entered boxes: {len(items_entered_boxes)}")
    elif entry_enabled and not items_entered_boxes:
        logger.warning("Entry detection was enabled but no entries were detected!")
        logger.warning("This could mean:")
        logger.warning("  1. No boxes were detected by YOLO")
        logger.warning("  2. Items didn't overlap boxes enough (try lowering overlap_threshold)")
        logger.warning("  3. Items didn't stay in boxes long enough (try lowering entry_threshold)")
        if config.get('entry_detection.require_motion', False):
            logger.warning("  4. require_motion=True but items were already in boxes (set to False)")

    csv_file.close()
    logger.info(f"Detection log saved to {detection_log_file}")

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    logger.info("="*60)
    logger.info("Pipeline Complete!")
    logger.info("="*60)


# ---------------- Runner ---------------- #
if __name__ == "__main__":
    videoPath = sys.argv[1] if len(sys.argv) > 1 else None
    main(videoPath)