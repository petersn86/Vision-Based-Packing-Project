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
import sys, os
import cv2
from pathlib import Path
import csv

def main(videoPath):

    # Setup directories
    framesDir         = "../data/frames"
    annotatedDir      = "../data/yolo_frames"
    final_output_file = "refined_item_list.txt"
    detection_log_file = "detection_log.csv"

    # Ensure annotated output directory exists (main.py previously didn't create it)
    Path(annotatedDir).mkdir(parents=True, exist_ok=True)

    # Extract frames (capture returned metadata to get timestamps)
    print(f"[INFO] Extracting frames from video: {videoPath}")
    frames_meta = extractFrames(videoPath, framesDir, 2.0)

    # Load frames
    framesData = loadFrames(framesDir)

    # Build mapping from filename -> timestamp (seconds)
    meta_map = {}
    try:
        for m in frames_meta:
            fname = os.path.basename(m.get('path', ''))
            meta_map[fname] = m.get('timestamp', None)
    except Exception:
        meta_map = {}

    # Object detector (YOLO models)
    detector = ObjectDetector(
        "../models/yolo11l.pt",
        "../models/cardboard_boxYOLO.pt"
    )

    # FrameReasoner without item list
    reasoner = FrameReasoner(model_name="llama3.2-vision")

    print("[INFO] Running YOLO detections and LLaMA refinement on frames...")

    # Prepare detection log CSV
    # Order goes frame_index, filename, timestamp_s, yolo_label, confidence, refined_label
    csv_file = open(detection_log_file, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_index", "filename", "timestamp_s", "yolo_label", "confidence", "refined_label"])

    finalItems = set()    # Collect unique refined labels
    ignoreLabels = {"person", "people", "cardboard", "box"}

    # Loop over frames
    for f in framesData:
        originalFrame = f["frame"]
        detections = detector.detectObjects(originalFrame)

        # Lookup timestamp for this frame (seconds)
        timestamp = meta_map.get(f.get('filename'), None)

        if not detections:
            continue

        print(f"\n--- Processing Frame {f['index']} ({len(detections)} detections) ---")

        # Loop through detections
        for det_idx, det in enumerate(detections):
            yoloLabel = det["label"]

            # Ignore people / cardboard box labels
            if yoloLabel.lower() in ignoreLabels:
                continue

            # Crop object from frame
            x1, y1, x2, y2 = det["bbox"]
            croppedImage = originalFrame[y1:y2, x1:x2]

            # Skip empty crop errors
            if croppedImage.size == 0:
                print(f"[WARNING] Skipping empty crop for label '{yoloLabel}' in frame {f['index']}")
                continue

            # Save crop with detection index and timestamp (if available) to avoid overwriting
            ts_tag = f"t{int(timestamp*1000)}" if timestamp is not None else "t_unknown"
            cropPath = os.path.join(
                annotatedDir,
                f"frame_{f['index']:04d}_{ts_tag}_det_{det_idx}.jpg"
            )
            cv2.imwrite(cropPath, croppedImage)

            # Refine with LLaMA
            refinedLabel = reasoner.refineDetection(croppedImage, yoloLabel)

            # Print results and log timestamped detection
            if refinedLabel:
                if refinedLabel not in finalItems:
                    print(f"  [NEW ITEM] {timestamp}s YOLO: '{yoloLabel}' → LLaMA: '{refinedLabel}'")
                    finalItems.add(refinedLabel)
                else:
                    print(f"  [CONFIRMED] {timestamp}s YOLO: '{yoloLabel}' → LLaMA: '{refinedLabel}'")
            else:
                print(f"  [DISCARDED] {timestamp}s LLaMA rejected YOLO label '{yoloLabel}'")

            # Write a row to the detection log (timestamp may be None)
            csv_writer.writerow([
                f.get('index'),
                f.get('filename'),
                timestamp,
                yoloLabel,
                det.get('confidence'),
                refinedLabel
            ])

    print("\n--- Processing Complete ---\n")

    # Sort and print final items
    finalItemsList = sorted(list(finalItems))
    print(f"Found {len(finalItemsList)} unique refined items:")
    for item in finalItemsList:
        print(f"- {item}")

    # Save results
    with open(final_output_file, 'w') as f:
        for name in finalItemsList:
            f.write(f"{name}\n")

    # Close CSV log
    csv_file.close()

    print(f"\n[INFO] Items saved to {final_output_file}")


# Runner
if __name__ == "__main__":
    if len(sys.argv) > 1:
        videoPath = sys.argv[1]
    else:
        # Default video path (can be changed)
        videoPath = "data/videos/packing_video.mp4"

    main(videoPath)
