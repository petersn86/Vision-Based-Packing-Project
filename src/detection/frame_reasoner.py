#######################################################
# @Author: Josh Tourigny
# @Contributor(s): Peter Nolan
# @Document: 'frame_reasoner.py'
#
# Description:
# This module uses Llama Vision to analyze an image crop
# from YOLO. It verifies or refines the YOLO detections
# and returns a single confirmed label.
#
# Box context: if a list of already-confirmed labels for
# the current box is provided, they are injected into the
# prompt so LLaMA anchors its response against known items,
# reducing label drift across frames for the same object.
#
########################################################

import ollama
import numpy as np
import cv2
from typing import Optional, List

# Labels that indicate a box/container — LLaMA should never return these
BOX_LIKE_LABELS = {
    "box", "boxes", "cardboard", "cardboard box", "carton", "cartons",
    "container", "containers", "crate", "crates", "bin", "bins",
    "package", "packages", "packaging"
}

class FrameReasoner:

    def __init__(self, model_name: str = "llama3.2-vision"):
        print(f"[INFO] Initializing Frame Reasoner for Ollama model: {model_name}")
        self.model_name = model_name
        self.client = None

        # Connect to Ollama
        try:
            ollama.show(self.model_name)
            print(f"[INFO] Successfully connected to Ollama & found model: {self.model_name}")
            self.client = ollama.Client()
        except Exception as e:
            print(f"[ERROR] Failed to connect to Ollama or find model '{self.model_name}'.")
            print("Please ensure the Ollama server is running and the model is pulled.")
            print(f"Ollama error: {e}")

    # --------------------------------------------------------------
    # Build prompt, optionally injecting confirmed box context
    # --------------------------------------------------------------
    def buildPrompt(self, box_context: List[str] = None) -> str:
        """
        Build the LLaMA prompt.

        Args:
            box_context: List of refined labels already confirmed in this box.
                         If provided, LLaMA is asked to match against them
                         before inventing a new label, reducing drift where
                         the same object gets slightly different names across
                         frames (e.g. "Hand Sanitizer" vs "Hand Soap").
        """
        context_line = ""
        if box_context:
            context_line = (
                f"Items already confirmed in this box: "
                f"{', '.join(box_context)}\n"
                f"If this object matches one of those exactly, return that "
                f"exact name. Otherwise return the correct name.\n\n"
            )

        return (
            "You are an object identifier. Identify the object in this image.\n\n"
            "YOLO detected: '{yolo_label}'\n\n"
            f"{context_line}"
            "RULES:\n"
            "1. Return ONLY the object name (1-3 words max)\n"
            "2. If YOLO is correct, return that exact name\n"
            "3. If YOLO is wrong, return the correct name\n"
            "4. NO explanations, NO sentences, NO extra text\n"
            "5. If unclear or not a packable item, return: none\n\n"
            "GOOD examples: backpack, laptop, water bottle, book\n"
            "BAD examples: The object is a backpack, I think this is...\n\n"
            "Object name (one word or short phrase only):"
        )

    # --------------------------------------------------------------
    # Check if a label refers to a box/container
    # --------------------------------------------------------------
    def _is_box_label(self, label: str) -> bool:
        """Return True if the label is a box or container type that should be ignored."""
        clean = label.strip().lower()
        if clean in BOX_LIKE_LABELS:
            return True
        for box_word in ("box", "carton", "cardboard", "container", "crate", "bin"):
            if box_word in clean:
                return True
        return False

    # --------------------------------------------------------------
    # Clean and normalize LLaMA output
    # --------------------------------------------------------------
    def parseResponse(self, response_text: str) -> Optional[str]:
        """Clean and normalize LLaMA output"""
        if not response_text:
            return None

        clean = response_text.strip().lower()

        patterns_to_remove = [
            "the object in the image is a ",
            "the object in the image is an ",
            "the object in the image is ",
            "the object is a ",
            "the object is an ",
            "the object is ",
            "this is a ",
            "this is an ",
            "this is ",
            "i see a ",
            "i see an ",
            "answer: ",
            "final answer: ",
            "object identification",
            "correct response is: ",
            "so the correct response is: ",
            "object name: ",
        ]

        for pattern in patterns_to_remove:
            clean = clean.replace(pattern, "")

        clean = clean.strip("*\"'.!? ")

        if '\n' in clean:
            clean = clean.split('\n')[0].strip()

        if '.' in clean:
            clean = clean.split('.')[0].strip()

        clean = clean.strip("*\"'.!?- ")

        if not clean or clean == "none" or clean == "unknown":
            return None

        if len(clean.split()) > 4:
            return None

        if self._is_box_label(clean):
            return None

        return clean.title()

    # --------------------------------------------------------------
    # Main reasoning call
    # --------------------------------------------------------------
    def refineDetection(self,
                        image_crop: np.ndarray,
                        yolo_label: str,
                        box_context: List[str] = None) -> Optional[str]:
        """
        Refine a YOLO detection label using LLaMA vision.

        Args:
            image_crop:  Cropped BGR image of the detected object
            yolo_label:  Raw YOLO class label
            box_context: Labels already confirmed in this box (for anchoring).
                         Pass registry.get_unique_labels_for_box(box_id) here.

        Returns:
            Refined label string, or yolo_label as fallback
        """
        if self.client is None:
            print("[ERROR] FrameReasoner not initialized (client unavailable).")
            return yolo_label

        try:
            _, buffer = cv2.imencode('.jpg', image_crop)
            image_bytes = buffer.tobytes()

            prompt_text = self.buildPrompt(box_context).format(yolo_label=yolo_label)

            messages = [
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [image_bytes]
                }
            ]

            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False
            )

            raw_response_text = response['message']['content']

            if len(raw_response_text) > 30:
                print(f"[WARNING] LLaMA verbose response for '{yolo_label}': "
                      f"{raw_response_text[:100]}...")

            parsed = self.parseResponse(raw_response_text)

            if parsed is not None and self._is_box_label(parsed):
                print(f"[INFO] LLaMA returned box-like label '{parsed}' "
                      f"for '{yolo_label}', using YOLO label")
                parsed = None

            if parsed is None:
                print(f"[INFO] LLaMA rejected '{yolo_label}', using YOLO label")
                return yolo_label

            return parsed

        except Exception as e:
            print(f"[ERROR] LLaMA refinement failed for '{yolo_label}': {e}")
            return yolo_label