##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'hand_detector.py'
#
# Description:
# Lightweight wrapper around the hands_weight.pt YOLO model.
# Returns a boolean indicating whether any hand-related detection
# was found in a given frame.
#
# The model uses PPE/safety classes:
#   'gloves', 'no_gloves', 'no_vest', 'no_hardhat'
# All of these imply a person's hands (and body) are visible,
# so any detection from this model is treated as hand_detected=True.
#
##############################################

import numpy as np
from ultralytics import YOLO
from typing import Optional


# All classes this model can produce — any hit means hands are present
HAND_MODEL_CLASSES = {'gloves', 'no_gloves', 'no_vest', 'no_hardhat'}


class HandDetector:
    """
    Runs hands_weight.pt on a frame and returns True if any
    hand-related detection is found above the confidence threshold.

    Intended to be called intermittently (every N frames) rather than
    on every frame, since it's used only for a metadata column in the
    detection log rather than driving tracking decisions.
    """

    def __init__(self,
                 model_path: str,
                 conf_threshold: float = 0.35):
        """
        Args:
            model_path:      Path to hands_weight.pt
            conf_threshold:  Minimum confidence to count a detection
        """
        print(f"[INFO] Loading hand detection model: {model_path}")
        self.model          = YOLO(model_path)
        self.conf_threshold = conf_threshold

        self._last_result: bool = False   # preserved for error fallback

        print(f"[INFO] HandDetector ready | conf={conf_threshold}")

    def detect(self, frame: np.ndarray) -> bool:
        """
        Return True if hands are detected in frame.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            bool — True if any hand-related class was detected
        """
        try:
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                verbose=False
            )

            for result in results:
                for box in result.boxes:
                    cls   = int(box.cls[0].cpu().numpy())
                    label = self.model.names.get(cls, '').lower().strip()
                    if label in HAND_MODEL_CLASSES:
                        self._last_result = True
                        return True

            self._last_result = False
            return False

        except Exception as e:
            print(f"[WARNING] HandDetector.detect() failed: {e}")
            # On error, preserve last known result rather than crashing
            return self._last_result
