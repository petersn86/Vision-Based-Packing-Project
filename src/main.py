##############################################
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
##############################################

<<<<<<< Updated upstream
from video_processor            import extractFrames
from detection.frame_loader     import loadFrames
from detection.object_detector  import ObjectDetector
import sys
=======
# main.py
from video_processor import extractFrames
from detection.frame_loader import loadFrames
from detection.object_detector import ObjectDetector, save_frame_with_boxes 
from detection.frame_reasoner import FrameReasoner
import sys, os
import cv2 
import shutil
>>>>>>> Stashed changes

# Setup driver function
def main(videoPath):
<<<<<<< Updated upstream
    frames      = extractFrames(videoPath, "../data/frames", 2.0) # might not need 'frames'
    framesData  = loadFrames("../data/frames")
    detector    = ObjectDetector("../models/yolo11m.pt", "../models/cardboard_boxYOLO.pt")

    detectedItems = set() #Set for printing unique items / Use '[]' instead of 'set()' for list of all items

    for f in framesData:
        detections = detector.detectObjects(f["frame"])
        print(f"Frame {f['index']} detections:", detections)   
        
        for detection in detections: #For each frame detection Add detected item names to set 
            detectedItems.add(detection['label']) # #Use detectedItems.append(detection['label']) for list of all items

    print(f"\nFinished processing. All unique items found: {detectedItems}") #Terminal list of items    
    outputFile = 'item_list.txt' #text file of items
    with open(outputFile, 'w') as f:
        for name in detectedItems: #Write all names in the file on a new line
            f.write(f"{name}\n")
    print(f"Successfully saved unique class names to {outputFile}") #List confirmation
=======
    frames_dir = "../data/frames" #where frames are saved
    annotated_dir = "../data/yolo_frames" #where YOLO frames go
    item_list_path = "item_list.txt" # The master list for Llama 
    final_output_file = "refined_item_list.txt" # The final list for Llama

    # Ensure output dirs exist
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    if not os.path.exists(item_list_path): #check for item list
        print(f"[ERROR] Master item list not found at: {item_list_path}")
        print("Please create this file with one item name per line.")
        sys.exit(1)

    if os.path.exists(frames_dir): #check for frames folder
        print(f"[INFO] Clearing old frames from: {frames_dir}")
        shutil.rmtree(frames_dir) #removing frames from folder so that it doesn't analyze past frames in the folder

    print(f"[INFO] Extracting frames from video: {videoPath}") # Extract frames
    extractFrames(videoPath, frames_dir, 2.0)
    
    framesData = loadFrames(frames_dir) # Load frames
    
    detector = ObjectDetector("../models/yolo11m.pt", "../models/cardboard_boxYOLO.pt") # Initialize Detector

    reasoner = FrameReasoner( # Initialize Llama vision reasoner
        item_list_path=item_list_path,
        model_name="llama3.2-vision"
    )

    print("[INFO] Running YOLO detection and Llama refinement on all frames...")

    final_refined_items = set() #Set for printing unique items
    
    ignore_labels = {"person", "people", "cardboard", "box"} # Skipping body and box labels

    for f in framesData:
        original_frame = f["frame"]
        detections = detector.detectObjects(original_frame)    
        if not detections:
            continue

        print(f"--- Processing Frame {f['index']} ({len(detections)} YOLO detections) ---")

        annotated_path = os.path.join(annotated_dir, f"frame_{f['index']:04d}_boxed.jpg") #annotated frame for debugging (can remove)
        save_frame_with_boxes(original_frame, detections, annotated_path)

        for det in detections: #Detection refinement for each detection
            yolo_label = det["label"]    
            if yolo_label.lower() in ignore_labels:
                continue

            # Crop the object from the frame
            x1, y1, x2, y2 = det['bbox']
            cropped_image = original_frame[y1:y2, x1:x2]

            
            if cropped_image.size == 0: # Ensure crop is valid
                print(f"[WARNING] Skipping empty crop for label '{yolo_label}' in frame {f['index']}")
                continue
         
            refined_label = reasoner.refine_detection(cropped_image, yolo_label) # Ask Llama to refine this specific crop
            
            if refined_label: # If Llama returned a valid, refined label, add it to our final set
                if refined_label not in final_refined_items:
                    print(f"  [NEW ITEM] YOLO ('{yolo_label}') -> Llama ('{refined_label}')")
                    final_refined_items.add(refined_label)
                else:
                    print(f"  [CONFIRMED] YOLO ('{yolo_label}') -> Llama ('{refined_label}')") # We've seen it before, but it's good to see the confirmation
            else:
                print(f"  [DISCARDED] Llama discarded YOLO label '{yolo_label}'") # Llama returned 'None' or an invalid response


    # Final Output
    print("\n--- Processing Complete ---")
    
    final_item_list = sorted(list(final_refined_items))  # Convert set to a sorted list 


    print(f"Finished processing. Found {len(final_item_list)} unique items:") #Terminal list of items    
    for item in final_item_list:
        print(f"- {item}")

    with open(final_output_file, 'w') as f: # Write final items to file
        for name in final_item_list: #Write all names in the file on a new line
            f.write(f"{name}\n")
    print(f"\nSuccessfully saved unique refined items to {final_output_file}")

>>>>>>> Stashed changes

if __name__ == "__main__":
    if len(sys.argv) > 1:
        videoPath = sys.argv[1]
<<<<<<< Updated upstream
    else: # ignore this for now
        videoPath = "data/videos/packing_video.mp4"  # default video path 
    main(videoPath)
=======
    else:
        videoPath = "data/videos/packing_video.mp4"  # default path
    main(videoPath)
>>>>>>> Stashed changes
