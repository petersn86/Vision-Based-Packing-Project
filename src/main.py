#############################################################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'main.py'
#
# Description:
# Main entry point for the Vision-Based-Packing-Project pipeline.
# Extracts frames, runs YOLO detections, then refines labels using LLaMA
# vision reasoning through Ollama.
###############################################################################

from video_processor            import extractFrames
from detection.frame_loader     import loadFrames
from detection.object_detector  import ObjectDetector
from detection.frame_reasoner   import FrameReasoner
from detection.object_tracker   import ObjectTracker
from detection.video_annotator  import VideoAnnotator
from detection.qr_detector      import QRDetector, BoxItemMapper
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
    logger.info("Vision-Based Packing Project - Integrated Pipeline")
    logger.info("="*60)

    # ---------------- Paths ---------------- #
    # FIX: Use config.get() not config.get_path() to avoid double directory creation
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
    # FIX: Convert Path to string for extractFrames
    frames_meta = extractFrames(str(videoPath), str(framesDir), frame_interval)

    # FIX: Convert Path to string for loadFrames
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
    
    logger.info(f"Initializing YOLO models: {yolo_model}, {box_model}")
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
            logger.info(f"  track_thresh={track_thresh}, track_buffer={track_buffer}")
            logger.info(f"  match_thresh={match_thresh}, second_match_thresh={second_match_thresh}")
            tracker = ObjectTracker(
                track_thresh=track_thresh or 0.5,
                track_buffer=track_buffer or 30,
                match_thresh=match_thresh or 0.8,
                second_match_thresh=second_match_thresh or 0.5
            )
        else:
            # Fallback to SORT-style params
            max_age = config.get('tracking.max_age', 30)
            min_hits = config.get('tracking.min_hits', 3)
            iou_threshold = config.get('tracking.iou_threshold', 0.3)
            logger.info(f"Object tracking enabled (SORT compatibility mode)")
            logger.info(f"  max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}")
            tracker = ObjectTracker(
                track_thresh=0.5,
                track_buffer=max_age,
                match_thresh=iou_threshold + 0.5,
                second_match_thresh=iou_threshold
            )
    else:
        logger.info("Object tracking disabled")

    # ---------------- QR Detection ---------------- #
    qr_enabled = config.get('qr_detection.enabled', True)
    qr_detector = None
    box_mapper = None
    if qr_enabled:
        proximity_threshold = config.get('qr_detection.proximity_threshold', 100)
        logger.info(f"QR code detection enabled (proximity_threshold={proximity_threshold})")
        qr_detector = QRDetector(proximity_threshold)
        box_mapper = BoxItemMapper()
    else:
        logger.info("QR code detection disabled")

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
        output_video_path = PROJECT_ROOT / config.get('paths.annotated_video_file', 'data/output_annotated.mp4')
        video_fps = config.get('output.video_fps', 10)
        video_codec = config.get('output.video_codec', 'mp4v')
        logger.info(f"Video annotation enabled (output: {output_video_path})")
        # FIX: Convert Path to string
        annotator = VideoAnnotator(str(output_video_path), fps=video_fps, codec=video_codec)

    logger.info("Running detection pipeline...")

    # ---------------- Detection Log ---------------- #
    csv_file = open(detection_log_file, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_index", "filename", "timestamp_s", "yolo_label", "confidence", "refined_label", "track_id", "box_id"])

    finalItems = set()
    ignoreLabels = set(config.get('detection.ignore_labels', []))
    processed_tracks = set()

    # ---------------- Main Frame Loop ---------------- #
    for f in framesData:
        originalFrame = f["frame"]
        frame_idx = f['index']
        timestamp = meta_map.get(f.get('filename'), None)

        detections = detector.detectObjects(originalFrame, confThresh=conf_threshold)

        # QR Codes
        qr_codes = []
        if qr_detector:
            qr_codes = qr_detector.detect_qr_codes(originalFrame)
            if qr_codes:
                qr_detector.update_boxes(qr_codes, frame_idx)
                logger.debug(f"Frame {frame_idx}: Detected {len(qr_codes)} QR codes")

        # Object Tracking
        tracked_objects = []
        if tracker:
            tracked_objects = tracker.update(detections)
            logger.debug(f"Frame {frame_idx}: {len(tracked_objects)} tracked objects")
        else:
            tracked_objects = [{**det, 'track_id': None} for det in detections]

        if video_output_enabled:
            detections_per_frame[frame_idx] = tracked_objects

        # Process each tracked object
        for obj in tracked_objects:
            track_id = obj.get('track_id')
            yoloLabel = obj["label"]
            confidence = obj["confidence"]
            bbox = obj["bbox"]

            if yoloLabel.lower() in ignoreLabels:
                continue

            if tracker and track_id is not None and track_id in processed_tracks:
                continue

            x1, y1, x2, y2 = bbox
            croppedImage = originalFrame[y1:y2, x1:x2]
            if croppedImage.size == 0:
                logger.warning(f"Skipping empty crop for '{yoloLabel}' in frame {frame_idx}")
                continue

            # Refine with LLaMA
            refinedLabel = None
            if reasoner:
                refinedLabel = reasoner.refineDetection(croppedImage, yoloLabel)
                if track_id is not None:
                    processed_tracks.add(track_id)
            else:
                refinedLabel = yoloLabel

            # Map to box
            box_id = None
            if qr_detector and refinedLabel:
                box_id = qr_detector.map_item_to_box(bbox, frame_idx)
                if box_id and box_mapper:
                    box_mapper.add_mapping(track_id or -1, refinedLabel, box_id, timestamp or 0.0)
                    logger.info(f"[MAPPED] Item '{refinedLabel}' -> Box '{box_id}'")

            # Log results
            if refinedLabel:
                if refinedLabel not in finalItems:
                    logger.info(f"[NEW ITEM] {timestamp}s Track#{track_id} '{yoloLabel}' -> '{refinedLabel}'")
                    finalItems.add(refinedLabel)
                else:
                    logger.debug(f"[CONFIRMED] {timestamp}s Track#{track_id} '{refinedLabel}'")
            else:
                logger.debug(f"[DISCARDED] {timestamp}s Rejected '{yoloLabel}'")

            csv_writer.writerow([frame_idx, f.get('filename'), timestamp, yoloLabel, confidence, refinedLabel, track_id, box_id])

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
    
    with open(final_output_file, 'w', encoding='utf-8') as f:
        for name in finalItemsList:
            f.write(f"{name}\n")
    
    logger.info(f"Items saved to {final_output_file}")

    if box_mapper:
        box_mappings = box_mapper.export_to_dict()
        box_mapping_file = PROJECT_ROOT / 'box_mappings.json'
        with open(box_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(box_mappings, f, indent=2)
        logger.info(f"Box mappings saved to {box_mapping_file}")
        logger.info(f"Total boxes detected: {box_mappings['summary']['total_boxes']}")
        logger.info(f"Total items mapped: {box_mappings['summary']['total_items']}")

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