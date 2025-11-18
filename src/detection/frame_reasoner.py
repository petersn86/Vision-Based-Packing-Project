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

import re

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
        
# Experimental function for ollama handling just the raw frames
    def rawFrameReasoning(self, framePaths) -> list:

        if self.client is None:
            print("[ERROR] Llama client not initialized.")
            return []
        
        if not framePaths:
            print("[ERROR] No frames provided.")
            return []
        
        imageBytes      = []

        frames = []
        for path in framePaths:
            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Could not load frame: {path}")
                continue
            frames.append(img)

        if not frames:
            print("[ERROR] No valid frames loaded.")
            return[]
        
        # -- Build an image collage --
        collage = self.make_collage(framePaths)

        if collage is None:
            print("[ERROR] Collage generation failed.")
            return []

        # Save debug image
        try:
            cv2.imwrite('../data/videos/collage.jpg', collage)
        except Exception as e:
            print(f"[ERROR] Failed to save collage: {e}")

        # Encode for Llama
        _, buffer = cv2.imencode('.jpg', collage)
        collageBytes = buffer.tobytes()

        # Build prompt
        prompt = (
            "You are analyzing frames from a video of objects being packed into a cardboard box. These frames were made into a collage with 4 frames in each row for your convenience.\n"
            "Analyze the collage of ALL of the provided images and identify the distinct physical items that are being packed into the box.\n"
            "Return only a clean list of item names, one per line. If there are multiple of a distinct object type being packed into the box, put the numerical amount next to the item name in the list."

        )

        messages = [
            {
                "role"   :  "user",
                "content":  prompt,
                "images" :  [collageBytes]
            }
        ]

        try:
            response = self.client.chat(
                model = self.model_name,
                messages=messages,
                stream=False
            )
            raw = response["message"]["content"]
            return self.parseRawList(raw)
        
        except Exception as e:
            print(f"[ERROR] Llama raw frame reasoning failed: {e}")
            return []

    # Experimental helper function
    def parseRawList(raw: str) -> dict:
        
        items = {}
        lines = [l.strip() for l in raw.split("\n") if l.strip()]

        for line in lines:
            cleaned = re.sub(r"^[\-\*\d\.\)\(]+\s*", "", line).strip().lower()

            patterns = [
                r"^(.*?)[\s\(]*x?(\d+)[\)]*$",     # apple x3 / apple (3) / apple 3
                r"^(\d+)\s+(.*)$",                 # 3 apples
            ]

            item_name = cleaned
            count = 1
            matched = False

            for p in patterns:
                m = re.match(p, cleaned)
                if m:
                    if p == patterns[0]:
                        item_name = m.group(1).strip()
                        count     = int(m.group(2))
                    else:
                        count     = int(m.group(1))
                        item_name = m.group(2).strip()
                    matched = True
                    break

            item_name = re.sub(r"s$", "", item_name)

            if item_name in items:
                items[item_name] += count
            else:
                items[item_name] = count

        return items
    
    def make_collage(self, framePaths, tile_size=(300,300), columns=3):

        images = []
        w, h = tile_size

        for path in framePaths:
            img = cv2.imread(path)

            if img is None:
                print(f"[WARN] Invalid frame skipped: {path}")
                continue

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img = cv2.resize(img, (w, h))

            images.append(img)

        if not images:
            print("[ERROR] No valid frames for collage.")
            return None

        rows = []
        for i in range(0, len(images), columns):
            row_imgs = images[i:i+columns]

            while len(row_imgs) < columns:
                row_imgs.append(np.zeros((h, w, 3), dtype=np.uint8))

            rows.append(cv2.hconcat(row_imgs))

        collage = cv2.vconcat(rows)

        max_width = 1024
        h2, w2 = collage.shape[:2]

        if w2 > max_width:
            scale = max_width / w2
            collage = cv2.resize(collage, (max_width, int(h2 * scale)))

        return collage
