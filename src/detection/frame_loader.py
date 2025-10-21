##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'frame_loader.py'
#
# Description:
# This module provides functions to load extracted video frames
# from a specified directory. Frames are read in sorted order and
# returned as a list of (frame, metadata) pairs for use in the
# Per-Frame Object Detection Module (PFODM).
#
##############################################

from typing import List, Dict
import os
import cv2


# Load frames from extracted video
def loadFrames(framesDir: str) -> List[Dict]:

    # Supported file formats
    supportedExts   = (".jpg", ".jpeg", ".png")

    # Get frames from directory
    frameFiles      = [ f for f in os.listdir(framesDir) if f.lower().endswith(supportedExts)]

    # Sort Numerically
    frameFiles.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

    framesData      = []

    # Image read frames from directory and store data
    for i, filename in enumerate(frameFiles):
        
        filePath    = os.path.join(framesDir, filename)
        frame       = cv2.imread(filePath)

        if frame is None:
            print(f"[Warning] Skipping unreadable frame: {filePath}")
            continue

        # Format dict
        framesData.append({
            "index"     : i,
            "filename"  : filename,
            "path"      : filePath,
            "frame"     : frame
        })
    
    
    print(f"[INFO] Loaded {len(framesData)} frames from {framesDir}")
    return framesData


