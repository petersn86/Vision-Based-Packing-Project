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
########################################################

import ollama
import numpy as np
import cv2
from typing import Optional

class FrameReasoner:

    def __init__(self, model_name: str = "llama3.2-vision"):
        print(f"[INFO] Initializing Frame Reasoner for Ollama model: {model_name}")
        self.model_name = model_name
        self.client = None

        # Build static prompt template
        self.prompt_template = self.buildPrompt()

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
    # Prompt for the LLM (no item list restrictions anymore!)
    # --------------------------------------------------------------
    def buildPrompt(self) -> str:
        return (
            "You are an expert object identifier.\n"
            "You are given a YOLO label: '{yolo_label}'.\n"
            "You are also given an image crop containing one object.\n\n"
            "Your task:\n"
            "- Identify the object in the image.\n"
            "- If the YOLO guess is correct, return the normalized name.\n"
            "- If YOLO is incorrect, return the corrected name.\n"
            "- If the object is unclear, not an item, a person, a cardboard box, "
            "or something unidentifiable, reply with EXACTLY: None\n\n"
            "Rules:\n"
            "- Return ONLY the final object name.\n"
            "- No extra words, no punctuation, no explanation.\n\n"
            "Final answer:"
        )

    # --------------------------------------------------------------
    # Clean and normalize LLaMA output
    # --------------------------------------------------------------
    def parseResponse(self, response_text: str) -> Optional[str]:
        clean_response = response_text.strip().strip(" -*\"().\n")

        if not clean_response:
            return None

        if clean_response.lower() == "none":
            return None

        # Optional: normalize synonyms (edit or remove as needed)
        synonyms = {
            "tape roll": "packing tape",
            "tape": "packing tape",
            "package tape": "packing tape",
        }
        if clean_response.lower() in synonyms:
            clean_response = synonyms[clean_response.lower()]

        return clean_response

    # --------------------------------------------------------------
    # Main reasoning call
    # --------------------------------------------------------------
    def refineDetection(self, image_crop: np.ndarray, yolo_label: str) -> Optional[str]:

        if self.client is None:
            print("[ERROR] FrameReasoner not initialized (client unavailable).")
            return None

        try:
            # Convert numpy image to JPG bytes
            _, buffer = cv2.imencode('.jpg', image_crop)
            image_bytes = buffer.tobytes()

            # Fill prompt
            prompt_text = self.prompt_template.format(yolo_label=yolo_label)

            # Build LLaMA request
            messages = [
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [image_bytes]
                }
            ]

            # Call Ollama Vision model
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False
            )

            raw_response_text = response['message']['content']

            # Parse output
            return self.parseResponse(raw_response_text)

        except Exception as e:
            print(f"[ERROR] LLaMA refinement failed for '{yolo_label}': {e}")
            return None
