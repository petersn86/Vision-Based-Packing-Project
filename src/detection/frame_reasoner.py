#######################################################
# @Author: Josh Tourigny
# @Contributor(s): Peter Nolan
# @Document: 'frame_reasoner.py'
#
# Description:
# This module uses Llama Vision to analyze an image crop
# from YOLO. It verifies the objects given from the 
# computer vision detections and returns a single,
# confirmed label.
#
########################################################

import ollama
import numpy as np
import cv2
from typing import Set, Optional

class FrameReasoner:

    # Initialize OOD
    def __init__(self, item_list_path: str, model_name: str = "llama3.2-vision"):
        print(f"[INFO] Initializing Frame Reasoner for Ollama model: {model_name}")
        self.model_name                         = model_name
        self.client                             = None

        # Load the master item list -- remove this?
        self.item_list, self.item_list_text     = self.loadItemList(item_list_path)
        if not self.item_list:
            print(f"[ERROR] Could not load items from {item_list_path}. Reasoner will not function.")
            return
        
        # Create the prompt for llama
        self.prompt_template                    = self.buildPrompt()

        # Connect to Ollama
        try:
            ollama.show(self.model_name)
            print(f"[INFO] Successfully connected to Ollama & found model: {self.model_name}")
            self.client                         = ollama.Client()
        except Exception as e:
            print(f"[ERROR] Failed to connect to Ollama or find model '{self.model_name}'.")
            print("Please ensure the Ollama server is running and you have pulled the model.")
            print(f"Ollama error: {e}")


    # Load in the master list of items -- remove this?
    def loadItemList(self, path: str) -> (Set[str], str):
        try:
            with open(path, 'r') as f:
                items = {line.strip().title() for line in f if line.strip()}

            if not items:
                return set(), ""
            
            items_list_text = "\n".join(f"- {item}" for item in sorted(list(items)))
            print(f"[INFO] Loaded {len(items)} items for Llama reasoning.")
            return items, items_list_text
        
        except FileNotFoundError:
            print(f"[ERROR] Item list not found: {path}")
            return set()

    # Builds the prompt to fed into the LLM
    def buildPrompt(self) -> str:
        return f""" You are an object identification expert.
        An object was detected in an image. The initial (YOLO) label is: '{{yolo_label}}'

        Analyze the provided image crop and refine this label.

        If the object in the image is clearly not any of these items, or if it is a person or a cardboard box, respond with 'None'.
        Only respond with the single, best-matching name from the list. Do not add any other text, explanation, or punctuation.
        Best match:"""

    # Cleans output and validates it with the master item list -- remove this?
    def parseResponse(self, response_text: str) -> Optional[str]:

        # Clean up artifacts
        clean_response = response_text.strip(" -*\"().\n")

        # Check for junk/ignore
        if not clean_response or "none" in clean_response.lower():
            return None
        
        # Looking for an exact item match
        for item in self.item_list:
            if item.lower() == clean_response.lower():
                return item
            
        # If an item is not in an item list, then it gets discarded. This may need to be removed
        print(f"[WARNING] Llama response '{clean_response}' is not in the master item list. Discarding...")
        return None
    
    def refineDetection(self, image_crop: np.ndarray, yolo_label: str) -> Optional[str]:

        if self.client is None or not self.item_list:
            return None # Reasoner not initialized correctly
        
        try:
            # Convert numpy image to bytes for the API
            _, buffer = cv2.imencode('.jpg', image_crop)
            image_bytes = buffer.tobytes()

            # Format the text prompt
            prompt_text = self.prompt_template.format(yolo_label= yolo_label)

            # Create the multimodal message payload
            messages = [
                {
                    'role'      : 'user',
                    'content'   : prompt_text,
                    'images'    : [image_bytes]

                }
            ]

            # Call Ollama
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False
            )

            raw_response_text = response['message']['content']

            # Parse and validate
            return self.parseResponse(raw_response_text)
        
        except Exception as e:
            print(f"[ERROR] Llama refinement failed for label '{yolo_label}' : {e}")
            return None