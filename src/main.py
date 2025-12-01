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

def main(videoPath):

    # Setup directories
    framesDir         = "../data/frames"
    annotatedDir      = "../data/yolo_frames"
    final_output_file = "refined_item_list.txt"

    # Extract frames
    print(f"[INFO] Extracting frames from video: {videoPath}")
    extractFrames(videoPath, framesDir, 2.0)

    # Load frames
    framesData = loadFrames(framesDir)

    # Object detector (YOLO models)
    detector = ObjectDetector(
        "../models/yolo11l.pt",
        "../models/cardboard_boxYOLO.pt"
    )

    # FrameReasoner without item list
    reasoner = FrameReasoner(model_name="llama3.2-vision")

    print("[INFO] Running YOLO detections and LLaMA refinement on frames...")

    finalItems = set()    # Collect unique refined labels
    ignoreLabels = {"person", "people", "cardboard", "box"}

    # Loop over frames
    for f in framesData:
        originalFrame = f["frame"]
        detections = detector.detectObjects(originalFrame)

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

            # Save crop with detection index to avoid overwriting
            cropPath = os.path.join(
                annotatedDir,
                f"frame_{f['index']:04d}_det_{det_idx}.jpg"
            )
            cv2.imwrite(cropPath, croppedImage)

            # Refine with LLaMA
            refinedLabel = reasoner.refineDetection(croppedImage, yoloLabel)

            # Print results
            if refinedLabel:
                if refinedLabel not in finalItems:
                    print(f"  [NEW ITEM] YOLO: '{yoloLabel}' → LLaMA: '{refinedLabel}'")
                    finalItems.add(refinedLabel)
                else:
                    print(f"  [CONFIRMED] YOLO: '{yoloLabel}' → LLaMA: '{refinedLabel}'")
            else:
                print(f"  [DISCARDED] LLaMA rejected YOLO label '{yoloLabel}'")

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

    print(f"\n[INFO] Items saved to {final_output_file}")


# Runner
if __name__ == "__main__":
    if len(sys.argv) > 1:
        videoPath = sys.argv[1]
    else:
        # Default video path (can be changed)
        videoPath = "data/videos/packing_video.mp4"

    main(videoPath)
