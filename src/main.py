#############################################################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'main.py'
#
# Description:
# This is the main entry point for the Vision-Based-Packing-Project pipeline.
# It orchestrates the workflow by calling video processing functions to extract
# frames from input videos, and optionally passes the frame data to downstream
# modules for item recognition and box content analysis. Designed to serve as
# the starting script for the full packing analysis process.
#
###############################################################################

from video_processor            import extractFrames
from detection.frame_loader     import loadFrames
from detection.object_detector  import ObjectDetector
from detection.frame_reasoner   import FrameReasoner
import sys, os
import cv2

# Setup driver function

def main(videoPath):

    # Setup variables
    framesDir           = "../data/frames"
    annotatedDir        = "../data/yolo_frames"
    item_list_path      = 'item_list.txt' # -- remove this ?
    final_output_file   = "refined_item_list.txt" # -- adjust this ?

    # Extract frames
    print(f"[INFO] Extracting frames from video: {videoPath}")
    extractFrames(videoPath, framesDir, 2.0)

    # Load frames
    framesData           = loadFrames(framesDir)

    # Create detector object for detections
    detector            = ObjectDetector("../models/yolo11l.pt", "../models/cardboard_boxYOLO.pt")

    # Load reasoner -- adjust item_list_path?
    reasoner            = FrameReasoner(
            item_list_path = item_list_path,
            model_name = "llama3.2-vision"
    )

    print("[INFO] Running YOLO detections and Llama refinement on all frames...")

    # Set for printing items
    finalItems          = set()

    # Ignore these labels
    ignoreLabels        = {"person", "people", "cardboard", "box"}

    # Iterate through the frame data
    for f in framesData:
        originalFrame   = f["frame"]
        detections      = detector.detectObjects(originalFrame) # Run YOLO on frame

        if not detections:
            continue

        print(f"--- Processing Frame {f['index']} ({len(detections)} YOLO detections) ---")

        # ------ DRAW BOUNDING BOX ------ #
        # annotatedFrame  = originalFrame.copy()
        # for det in detections:
        #     label       = det["label"]
        #     conf        = det.get("confidence", 0)
        #     x1,y1,x2,y2 = det["bbox"]
        #     color       = (0, 255, 0)
        #     cv2.rectangle(annotatedFrame, (x1,y1), (x2,y2), color, 2)
        #     cv2.putText(
        #         annotatedFrame,
        #         f"{label} {conf:.2f}",
        #         (x1, y1 - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.6,
        #         color,
        #         2
        #     )
        
        # annotatedPath   = os.path.join(annotatedDir, f"frame_{f['index']:04d}.jpg")
        # cv2.imwrite(annotatedPath, annotatedFrame)
        # --------------------------------- #

        for det in detections:
            yoloLabel       = det["label"]
            if yoloLabel.lower() in ignoreLabels:
                continue

            # Crop the object from the frame
            x1,y1,x2,y2     = det['bbox']
            croppedImage    = originalFrame[y1:y2, x1:x2]
            annotatedPath   = os.path.join(annotatedDir, f"frame_{f['index']:04d}.jpg")
            cv2.imwrite(annotatedPath, croppedImage)

            if croppedImage.size == 0:
                print(f"[WARNING] Skipping empty crop for label '{yoloLabel}' in frame {f['index']}")
                continue

            refinedLabel    = reasoner.refineDetection(croppedImage, yoloLabel)

            # Handle reasoned items
            if refinedLabel:
                if refinedLabel not in finalItems: # Doing only unique objects, fix this ?
                    print(f"     [NEW ITEM] YOLO ('{yoloLabel}') -> Llama ('{refinedLabel}')")
                    finalItems.add(refinedLabel)
                else:
                    print(f"     [CONFIRMED] YOLO ('{yoloLabel}') -> Llama ('{refinedLabel}')")

            else:
                print(f" [DISCARDED] Llama discared YOLO label '{yoloLabel}'")

    print("\n--- Processing Complete ---")

    finalItemsList      = sorted(list(finalItems))
    print(f"Finished processing. Found {len(finalItemsList)} unique items:")
    for item in finalItemsList:
        print(f"- {item}")

    with open(final_output_file, 'w') as f:
        for name in finalItemsList:
            f.write(f"{name}\n")
    print(f"\nSuccessfully saved items to {final_output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        videoPath = sys.argv[1]
    else: # ignore this for now
        videoPath = "data/videos/packing_video.mp4" # default video path
    main(videoPath)