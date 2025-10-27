##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'video_processor.py'
#
# Description:
# This module handles the extraction of frames from a video file
# at user-defined **time intervals (in seconds)**. Using OpenCV,
# it reads the input video, calculates frame indices based on
# the interval duration, and saves those frames as image files
# for later analysis. The function returns structured metadata
# for each saved frame, including its index, timestamp, and
# file path, which can be used in downstream computer vision
# or language model processing. Designed to be called by 'main.py'
# as part of the Vision-Based-Packing-Project pipeline.
#
##############################################

from pathlib import Path
from tqdm import tqdm
import os
import cv2

def extractFrames(videoPath: str, outPath: str = "data/frames", timeSkip: float = 1.0):

    # Ensure output directory exists
    Path(outPath).mkdir(parents=True, exist_ok=True)

    # Open video file
    vid             = cv2.VideoCapture(videoPath)
    if not vid.isOpened():
        raise ValueError(f"Could not open video file: {videoPath}")

    # Get frame count & FPS
    frameCount      = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps             = vid.get(cv2.CAP_PROP_FPS)
    duration        = frameCount / fps if fps > 0 else 0

    print(f"[INFO] Processing video: {videoPath}")
    print(f"[INFO] Total Frames: {frameCount}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
    print(f"[INFO] Extracting one frame every {timeSkip} second(s)...")

    # Calculate frame interval based on timeSkip
    frameInterval       = int(fps * timeSkip)
    if frameInterval    <= 0:
        frameInterval   = 1

    data                = []
    frameIndex          = 0
    savedCount          = 0

    with tqdm(total=frameCount, desc="Extracting Frames", unit="frame") as pbar:
        while True:
            ret, frame  = vid.read()
            if not ret:
                break

            if frameIndex % frameInterval == 0:
                timestamp = frameIndex / fps if fps > 0 else 0
                frameName = os.path.join(outPath, f"frame_{frameIndex:05d}.jpg")
                cv2.imwrite(frameName, frame)

                data.append({
                    "path": frameName,
                    "index": frameIndex,
                    "timestamp": round(timestamp, 3)
                })
                savedCount += 1

            frameIndex += 1
            pbar.update(1)

    vid.release()
    print(f"[INFO] Extraction complete. Saved {savedCount} frame(s) to '{outPath}'")

    return data
