##############################################
# @Author: Evan Rapoza
# @Contributor(s): 
# @Document: 'YoloVideoTesting.py'
#
# Description:
# This module provides real-time object detection using a live webcam feed.
# Using OpenCV (cv2) to access the default camera and the Ultralytics
# library, it loads a pre-trained YOLOv11 model (yolo11m.pt). The script
# captures frames from the webcam, runs YOLOv11 prediction on each frame,
# and displays the annotated video stream with bounding boxes in a
# window. This script serves as the live vision component for the
# Vision-Based-Packing-Project, identifying items in real-time.
#
##############################################

from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv11 model (models are n,s,m,l)
model = YOLO('yolo11m.pt')

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLOv8 prediction on the frame
        results = model.predict(frame)

        # Get the annotated frame with bounding boxesq
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv11 Webcam", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()