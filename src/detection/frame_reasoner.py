import cv2
import re
import numpy as np
import ollama
from tqdm import tqdm

class FrameReasoner:

    # ----------------------------------------------
    # Initialize Llama client and model
    # ----------------------------------------------
    def __init__(self, model_name="llama3.2-vision"):
        self.client = None
        self.model_name = model_name
        try:
            ollama.show(self.model_name)
            print(f"[INFO] Successfully connected to Ollama & found model: {self.model_name}")
            self.client                         = ollama.Client()
        except Exception as e:
            print(f"[ERROR] Failed to connect to Ollama or find model '{self.model_name}'.")
            print("Please ensure the Ollama server is running and you have pulled the model.")
            print(f"Ollama error: {e}")

    # ------------------------------------------------------------
    # FRAME-BY-FRAME REASONING (no collage, no master list)
    # ------------------------------------------------------------
    # - Each frame is sent individually to the model
    # - Extract items from each frame
    # - Accumulate totals across all frames
    # ------------------------------------------------------------
    # --- Frame-by-frame reasoning ---
    def reason_frame_by_frame(self, framePaths):

        if self.client is None:
            print("[ERROR] Llama client not initialized.")
            return []

        if not framePaths:
            print("[ERROR] No frames provided.")
            return []

        print(f"[INFO] Running frame-by-frame reasoning on {len(framePaths)} frames...")

        results = []

        # Loop over every frame with a progress bar
        for path in tqdm(framePaths, desc="Reasoning", unit="frame"):

            # Load image
            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Invalid frame: {path}")
                continue

            # Encode to JPG for Ollama
            ok, buffer = cv2.imencode(".jpg", img)
            if not ok:
                print(f"[WARN] Failed to encode frame: {path}")
                continue

            imgBytes = buffer.tobytes()

            # Build prompt
            prompt = (
                "You are analyzing a single frame from a video of objects being packed into a box.\n"
                "List ONLY the objects that appear in THIS frame. Do not guess.\n"
                "Return a clean list with one item per line.\n"
            )

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [imgBytes]
                }
            ]

            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False
                )

                raw = response["message"]["content"]
                parsed = self._parse_frame_list(raw)

                results.append(parsed)

            except Exception as e:
                print(f"[ERROR] Llama frame reasoning failed on '{path}': {e}")

        print("[INFO] Frame-by-frame reasoning complete.")
        return results



    # --- Parser Helper ---
    def _parse_frame_list(self, raw: str) -> list:

        items = []
        lines = [l.strip() for l in raw.split("\n") if l.strip()]

        for line in lines:
            # remove bullet points, numbering, etc.
            cleaned = re.sub(r"^[\-\*\d\.\)\(]+\s*", "", line).strip().lower()
            if cleaned:
                items.append(cleaned)

        return items