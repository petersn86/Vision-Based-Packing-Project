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

from video_processor            import extractFrames
from detection.frame_loader     import loadFrames
from detection.object_detector  import ObjectDetector
import sys

# Setup driver function
def main(videoPath):
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        videoPath = sys.argv[1]
    else: # ignore this for now
        videoPath = "data/videos/packing_video.mp4"  # default video path 
    main(videoPath)
