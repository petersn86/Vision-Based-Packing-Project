#######################################################
# @Author: Josh Tourigny
# @Contributor(s): Peter Nolan
# @Document: 'frame_reasoner.py'
#
# IMPROVED VERSION - Fixes label drift problem
#
# Changes:
# 1. Better LLaMA prompt that encourages label reuse
# 2. Added hierarchical label normalization
# 3. More explicit instructions about matching existing items
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

# Hierarchical label mapping - maps specific labels to generic categories
# This reduces label drift by normalizing related items to the same name
LABEL_HIERARCHY = {
    'Bottle': [
        'bottle', 'water bottle', 'hand sanitizer', 'hand soap', 
        'soap bottle', 'lotion bottle', 'shampoo bottle', 'sanitizer',
        'body wash', 'dish soap', 'cleaning spray', 'spray bottle'
    ],
    'Phone': [
        'phone', 'smartphone', 'cell phone', 'mobile phone', 'cellphone',
        'iphone', 'android', 'mobile', 'telephone'
    ],
    'Calculator': [
        'calculator', 'calc', 'adding machine'
    ],
    'Book': [
        'book', 'notebook', 'journal', 'textbook', 'novel', 'diary',
        'planner', 'agenda', 'notepad'
    ],
    'Remote': [
        'remote', 'remote control', 'tv remote', 'controller'
    ],
    'Pen': [
        'pen', 'ballpoint', 'marker', 'highlighter', 'sharpie'
    ],
    'Scissors': [
        'scissors', 'shears', 'snips'
    ],
    'Tape': [
        'tape', 'duct tape', 'masking tape', 'packing tape', 'scotch tape',
        'adhesive tape'
    ],
    'Charger': [
        'charger', 'phone charger', 'cable', 'charging cable', 'power cord',
        'usb cable', 'adapter'
    ],
    'Headphones': [
        'headphones', 'earbuds', 'earphones', 'airpods', 'headset'
    ],
    'Glasses': [
        'glasses', 'eyeglasses', 'sunglasses', 'spectacles', 'reading glasses'
    ],
    'Watch': [
        'watch', 'wristwatch', 'smartwatch', 'timepiece'
    ],
    'Wallet': [
        'wallet', 'purse', 'billfold', 'cardholder'
    ],
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
    # Build prompt with strong emphasis on reusing existing labels
    # --------------------------------------------------------------
    def buildPrompt(self, yolo_label: str, box_context: List[str] = None) -> str:
        """
        Build the LLaMA prompt with emphasis on label consistency.

        Args:
            yolo_label: The YOLO detection label
            box_context: List of refined labels already confirmed in this box.
                         LLaMA is strongly encouraged to reuse these labels
                         for similar objects to prevent duplicate entries.
        """
        context_line = ""
        if box_context:
            formatted_items = '\n'.join(f'  {i+1}. {item}' for i, item in enumerate(box_context))
            context_line = (
                f"══════════════════════════════════════\n"
                f"ITEMS ALREADY IN THIS BOX:\n"
                f"{formatted_items}\n"
                f"══════════════════════════════════════\n\n"
                f"⚠️  IMPORTANT:\n"
                f"If this object is CLEARLY the same item as something in the list,\n"
                f"return that exact name for consistency.\n\n"
                f"However, if this is a DIFFERENT object, return the correct new name,\n"
                f"even if YOLO thinks it's the same as something in the list.\n\n"
                f"Examples:\n"
                f"  • You see another water bottle, 'Bottle' is in list → return 'Bottle' ✓\n"
                f"  • You see a screwdriver, YOLO says 'bottle', 'Bottle' is in list → return 'Screwdriver' ✓\n"
                f"  • You see a phone, YOLO says 'cell phone', 'Calculator' is in list → return 'Phone' ✓\n\n"
                f"Trust your visual analysis FIRST. Only reuse names when truly the same.\n\n"
            )

        return (
            "You are an object identifier for a packing inventory system.\n"
            "Your job: identify what is in this image with a SHORT, CONSISTENT label.\n\n"
            f"{context_line}"
            f"YOLO detected: '{yolo_label}'\n\n"
            "RULES:\n"
            "1. Return ONLY the object name (1-3 words max)\n"
            "2. If object matches an existing box item AND is truly the same → return THAT NAME exactly\n"
            "3. If YOLO is correct and item not in box yet → return YOLO label\n"
            "4. If YOLO is wrong → return the correct name (ignore YOLO!)\n"
            "5. NO explanations, NO sentences, NO extra text\n"
            "6. If unclear or not packable → return: none\n\n"
            "IMPORTANT DISTINCTIONS:\n"
            "  • Screwdriver ≠ Bottle (even if cylindrical handle)\n"
            "  • Calculator ≠ Phone (different devices)\n"
            "  • Remote ≠ Phone (different functions)\n"
            "  • Tool handle ≠ Bottle (tools are not bottles)\n\n"
            "GOOD RESPONSES: Bottle, Phone, Book, Calculator, Apple, Screwdriver, Scissors\n"
            "BAD RESPONSES: The bottle appears to be, I think this is, It looks like\n\n"
            "Object name:"
        )

    # --------------------------------------------------------------
    # Label normalization - maps variations to canonical forms
    # --------------------------------------------------------------
    def normalize_label(self, label: str) -> str:
        """
        Normalize label to canonical form to reduce drift.
        
        Maps specific variations (e.g., "Hand Sanitizer", "Water Bottle") 
        to generic categories (e.g., "Bottle") to prevent duplicate entries.
        
        SAFE VERSION: Only normalizes exact matches or clear word patterns
        to prevent "screwdriver" from becoming "bottle".
        
        Args:
            label: Raw label from LLaMA
            
        Returns:
            Normalized label (generic category if matched, else original)
        """
        if not label:
            return label
            
        label_lower = label.lower().strip()
        
        # Check each category
        for generic, specifics in LABEL_HIERARCHY.items():
            for specific in specifics:
                # SAFE MATCHING: Only normalize if it's truly a variant
                
                # Method 1: Exact match
                if label_lower == specific:
                    print(f"[NORMALIZE] '{label}' → '{generic}' (exact match: {specific})")
                    return generic
                
                # Method 2: Multi-word match - all words from specific must be in label
                # "hand sanitizer" matches "hand" in specifics
                # "screwdriver" does NOT match "bottle" (different words)
                specific_words = set(specific.split())
                label_words = set(label_lower.split())
                
                if specific_words and specific_words.issubset(label_words):
                    print(f"[NORMALIZE] '{label}' → '{generic}' (word match: {specific})")
                    return generic
                
                # Method 3: Safe suffix/prefix patterns
                # "water bottle" ends with "bottle" → OK
                # "screwdriver" contains "bottle" as substring → NOT OK
                if len(specific.split()) == 1:  # Single word specific
                    # Only match if specific is a complete word in label
                    if f" {specific} " in f" {label_lower} ":
                        print(f"[NORMALIZE] '{label}' → '{generic}' (word boundary: {specific})")
                        return generic
                    # Or at start/end
                    if label_lower.startswith(specific + " ") or label_lower.endswith(" " + specific):
                        print(f"[NORMALIZE] '{label}' → '{generic}' (edge word: {specific})")
                        return generic
        
        # No match - keep original
        return label

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
            box_context: Labels already confirmed in this box (for consistency).
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

            prompt_text = self.buildPrompt(yolo_label, box_context)

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