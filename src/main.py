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
    frames      = extractFrames(videoPath, "../data/frames") # might not need 'frames'
    framesData  = loadFrames("../data/frames")
    detector    = ObjectDetector()

    for f in framesData:
        detections = detector.detectObjects(f["frame"])
        print(f"Frame {f['index']} detections:", detections)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        videoPath = sys.argv[1]
    else: # ignore this for now
        videoPath = "data/videos/packing_video.mp4"  # default video path
    main(videoPath)
