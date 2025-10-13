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

from video_processor import extractFrames
import sys

# Setup driver function
def main(videoPath):
    frames = extractFrames(videoPath, "../data/frames")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        videoPath = sys.argv[1]
    else: # ignore this for now
        videoPath = "data/videos/packing_video.mp4"  # default video path
    main(videoPath)
