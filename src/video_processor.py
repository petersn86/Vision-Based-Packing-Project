##############################################
# @Author: Peter Nolan
# @Contributor(s): 
# @Document: 'video_processor.py'
#
# Description:
# This module handles the extraction of frames from a video file
# at user-defined intervals. Using OpenCV, it reads the input video,
# skips a configurable number of frames, and saves selected frames
# as image files for later analysis. The function returns structured
# metadata for each saved frame, including its index, timestamp, and
# file path, which can be used in downstream computer vision or
# language model processing. Designed to be called by 'main.py' as
# part of the Vision-Based-Packing-Project pipeline.
#
##############################################

from pathlib import Path
from tqdm import tqdm
import os
import cv2

def extractFrames(videoPath: str, outPath: str = "data/frames", frameSkip = 1):

    # Check if output directory exists
    Path(outPath).mkdir(parents=True, exist_ok=True)

    # Open video file
    vid         = cv2.VideoCapture(videoPath)

    # Get amount of frames & FPS from video
    frameCount  = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = vid.get(cv2.CAP_PROP_FPS)

    # Data Structure containing extracted frame info
    data        = []

    # Display Info
    print(f"[INFO] Processing video: {videoPath}")
    print(f"[INFO] Total Frames: {frameCount}, FPS: {fps:.2f}")
    print(f"[INFO] Extracting every {frameSkip} frames...")
    
    # Setup variables for extraction
    frameIndex  = 0
    savedCount  = 0

    with tqdm(total=frameCount, desc="Extracing Frames", unit="frame") as pbar:
        while True:
            # Run until out of frames
            ret, frame = vid.read()
            if not ret:
                break
        
            # If we should, extract frame
            if frameIndex % frameSkip == 0:
                if fps > 0:
                    timestamp = frameIndex / fps 
                else:
                    timestamp = 0
                frameName = os.path.join(outPath, f"frame_{frameIndex:05d}.jpg")
                cv2.imwrite(frameName, frame)

                # Write frame info to data structure
                data.append({
                    "path": frameName,
                    "index": frameIndex,
                    "timestamp": round(timestamp, 3)
                })
                savedCount += 1

            frameIndex += 1
            pbar.update(1)

    # Finish & Return Data Structure
    vid.release()
    print(f"[INFO] Extraction complete. Saved {savedCount} frame(s) to '{outPath}'")

    return data




