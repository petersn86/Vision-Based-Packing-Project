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
            "You are an object identifier. Identify the object in this image.\n\n"
            "YOLO detected: '{yolo_label}'\n\n"
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
    # Clean and normalize LLaMA output
    # --------------------------------------------------------------
    def parseResponse(self, response_text: str) -> Optional[str]:
        """Clean and normalize LLaMA output"""
        if not response_text:
            return None
        
        # Convert to lowercase for processing
        clean = response_text.strip().lower()
        
        # Remove common verbose patterns
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
        
        # Remove asterisks, quotes, periods
        clean = clean.strip("*\"'.!? ")
        
        # Take only first line (ignore explanations)
        if '\n' in clean:
            clean = clean.split('\n')[0].strip()
        
        # Take only first sentence (if multiple)
        if '.' in clean:
            clean = clean.split('.')[0].strip()
        
        # Remove leading/trailing punctuation again
        clean = clean.strip("*\"'.!?- ")
        
        # Check for rejection
        if not clean or clean == "none" or clean == "unknown":
            return None
        
        # Reject if too long (likely an explanation)
        if len(clean.split()) > 4:
            return None
        
        # Optional: Handle "cardboard box" -> reject it
        if "cardboard" in clean and "box" in clean:
            return None
        
        # Capitalize properly for output
        return clean.title()

    # --------------------------------------------------------------
    # Main reasoning call
    # --------------------------------------------------------------
    def refineDetection(self, image_crop: np.ndarray, yolo_label: str) -> Optional[str]:

        if self.client is None:
            print("[ERROR] FrameReasoner not initialized (client unavailable).")
            return yolo_label

        try:
            _, buffer = cv2.imencode('.jpg', image_crop)
            image_bytes = buffer.tobytes()

            prompt_text = self.prompt_template.format(yolo_label=yolo_label)

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
            
            # Debug logging
            if len(raw_response_text) > 30:
                print(f"[WARNING] LLaMA verbose response for '{yolo_label}': {raw_response_text[:100]}...")
            
            parsed = self.parseResponse(raw_response_text)
            
            # Fallback to YOLO if parsing fails
            if parsed is None:
                print(f"[INFO] LLaMA rejected '{yolo_label}', using YOLO label")
                return yolo_label
                
            return parsed

        except Exception as e:
            print(f"[ERROR] LLaMA refinement failed for '{yolo_label}': {e}")
            return yolo_label
