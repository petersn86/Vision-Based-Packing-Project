# 📦 Vision-Based Packing Project

AI-powered system that watches packing footage and automatically produces a structured inventory list — no manual data entry required.

**Contributors:** Peter Nolan · Evan Rapoza · Joshua Tourigny · Max Manjos  
**Course:** Senior Design 2025 — Roger Williams University

---

## How It Works

1. You upload a video of items being packed into boxes through the web UI
2. YOLO detects every item and the cardboard box in each frame
3. ByteTrack assigns stable IDs to each object across frames
4. LLaMA 3.2 Vision refines ambiguous labels using visual context
5. The system detects when each item enters the box, and flags removals for your confirmation
6. A structured packing list is produced — grouped by box, downloadable in multiple formats

---

## Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12 | Must be on your system PATH |
| NVIDIA GPU + CUDA | Strongly recommended; CPU fallback is available but slow |
| [Ollama](https://ollama.com) | Local LLM inference server |
| pyzbar (Windows) | Included in the Windows wheel; requires Visual C++ Redistributable |

### 1 — Install Ollama and pull the model

```bash
# After installing Ollama from https://ollama.com
ollama pull llama3.2-vision
```

### 2 — Clone the repo and set up the environment

```bash
git clone https://github.com/your-org/vision-packing-project.git
cd vision-packing-project

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3 — Add model weights

Place the YOLO weight files in the `models/` directory:

```
models/
  yolo11l.pt              ← general-purpose item detector
  cardboard_YOLO25.pt     ← cardboard box detector
  hands_weight.pt         ← hand detector (for exit detection)
```

> Model weights are not included in this repository due to file size. Contact the project team or your instructor for the download link.

### 4 — Start the web server

```bash
python app.py
```

Open your browser and go to **http://localhost:5000**

---

## Using the Web Interface

### Uploading a video

1. Drag your packing video onto the upload area, or click **Select Video**
2. Supported formats: **MP4, AVI, MOV, MKV** — maximum size 500 MB
3. Click **Upload** — processing starts automatically in the background

### While processing

A progress indicator shows the current pipeline stage. Processing time depends on video length and GPU speed — a 2-minute video typically takes 3–5 minutes on an RTX 3060.

**Exit confirmation cards** will pop up in the bottom-right corner if the system detects an item may have been removed from a box. Review the photo and click:
- ✅ **Confirmed** — the item was removed; it will be marked as such in the output
- ❌ **False Alarm** — the item is still in the box; dismiss the alert

### Downloading results

Once complete, four download buttons appear:

| Button | File | Description |
|---|---|---|
| 📄 Item List | `refined_item_list.txt` | Human-readable list grouped by box |
| 📊 Detection Log | `detection_log.csv` | Every detection event with full metadata |
| 🗂 Full Registry | `item_registry.json` | Structured JSON with instance IDs, timestamps, status |
| 🎬 Annotated Video | `output_annotated.mp4` | Original footage with bounding boxes overlaid |

---

## Running Without the Web UI

You can run the pipeline directly from the command line:

```bash
cd src
python main.py path/to/your_video.mp4
```

Output files are written to the project root:
- `refined_item_list.txt`
- `item_registry.json`
- `box_mappings.json`
- `detection_log.csv`
- `data/videos/output_annotated.mp4`
- `data/yolo_frames/` — cropped JPEG images of each confirmed detection

---

## Box Identification

The system supports two modes for assigning items to boxes:

**Barcode / QR mode (default):** Attach a CODE128 barcode or QR code sticker to each box. The scanner reads the code and uses its value as the box ID throughout the pipeline. This is the recommended mode for multi-box packing scenarios.

**Auto-ID fallback:** If no barcode is detected, the system assigns IDs automatically (`BOX-001`, `BOX-002`, …). Sufficient for single-box scenarios.

To disable barcode scanning entirely, set `barcode_scanner.enabled: false` in `config.yaml`.

---

## Configuration

All settings are in **`config.yaml`** in the project root. No code changes are needed for typical adjustments. Key settings:

```yaml
video:
  frame_interval: 1.0         # seconds between extracted frames
                               # increase for speed, decrease for accuracy

detection:
  confidence_threshold: 0.50  # YOLO confidence cutoff (0.35–0.70 typical)
  yolo_model: models/yolo11l.pt
  box_model:  models/cardboard_YOLO25.pt

llama:
  enabled: true               # set false to skip LLaMA (faster, raw YOLO labels)
  model_name: llama3.2-vision

barcode_scanner:
  enabled: true               # set false to use auto-generated box IDs

entry_detection:
  overlap_threshold: 0.20     # fraction of item bbox that must overlap the box
  entry_threshold: 1          # consecutive frames inside box to confirm entry

exit_detection:
  hand_overlap_threshold: 0.50  # hand-to-item overlap to arm Stage 1
  absence_threshold: 5          # frames absent (after hand) before confirmation prompt
  geometric_threshold: 25       # frames absent (no hand) before confirmation prompt
```

---

## Project Structure

```
vision-packing-project/
│
├── app.py                    # Flask web server — start here
├── config.yaml               # All runtime configuration
├── requirements.txt          # Python dependencies
├── cleanup.py                # Deletes extracted frames between runs
│
├── src/
│   ├── main.py               # Pipeline orchestrator
│   ├── video_processor.py    # Frame extraction
│   ├── config_loader.py      # YAML config manager
│   └── detection/
│       ├── object_detector.py    # Dual YOLO inference
│       ├── object_tracker.py     # ByteTrack + MobileNetV2 re-ID
│       ├── frame_reasoner.py     # LLaMA 3.2 Vision label refinement
│       ├── item_registry.py      # Deduplication + instance tracking
│       ├── entry_detector.py     # Box entry detection
│       ├── exit_detector.py      # Two-stage exit detection
│       ├── hand_detector.py      # Hand detection for exit arming
│       ├── barcode_scanner.py    # 1D/QR barcode scanning (pyzbar)
│       ├── box_tracker.py        # IoU-based box ID propagation
│       ├── plausibility_filter.py # Size-based detection filter
│       ├── image_preprocessor.py  # CLAHE / sharpen / denoise
│       ├── frame_loader.py       # Load extracted frames from disk
│       └── video_annotator.py    # Annotated output video writer
│
├── models/                   # YOLO weight files (not in repo)
├── data/
│   ├── frames/               # Extracted frames (auto-cleared, not in repo)
│   ├── yolo_frames/          # Cropped detection images (not in repo)
│   ├── videos/               # Input + annotated output video (not in repo)
│   └── exit_crops/           # Exit confirmation thumbnails
├── templates/
│   └── index.html            # Flask UI
└── docs/                     # Documentation and test scripts
```

---

## Tech Stack

| Component | Library / Model |
|---|---|
| Object detection | [Ultralytics YOLO11L](https://github.com/ultralytics/ultralytics) + custom cardboard model |
| Multi-object tracking | ByteTrack + MobileNetV2 re-ID (PyTorch) |
| Label refinement | LLaMA 3.2 Vision via [Ollama](https://ollama.com) |
| Barcode / QR scanning | pyzbar |
| Image processing | OpenCV, NumPy |
| Web interface | Flask, flask-cors |
| Configuration | PyYAML |

---

## Troubleshooting

**"Ollama connection refused" error**  
Make sure Ollama is running. On Windows it starts automatically after installation, but you can also start it manually: open a terminal and run `ollama serve`.

**Items not being detected**  
Try lowering `detection.confidence_threshold` to `0.35` in `config.yaml`. Also ensure the box is clearly visible in the frame and the camera angle is reasonably overhead.

**Annotated video not appearing**  
Check that your GPU supports the `mp4v` codec. If not, try changing `output.video_codec` to `XVID` in `config.yaml`.

**pyzbar import error on Windows**  
Install the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and ensure the zbar DLL is on your PATH. The pyzbar Windows wheel includes the DLL, but it occasionally needs to be registered manually.

**Processing is very slow**  
Set `llama.enabled: false` in `config.yaml` to skip LLaMA refinement. Also increase `video.frame_interval` (e.g. `2.0` or `3.0`) to process fewer frames.

**Cropped item images not appearing in `data/yolo_frames/`**  
Ensure `output.save_cropped_images: true` is set in `config.yaml`. This was a known bug in earlier versions that has been fixed in the current release.

---

## Output File Reference

### `refined_item_list.txt`
```
[BOX-001]
  - Scissors
  - Bottle
  - Mug
  - Knife (removed)
```

### `item_registry.json`
```json
{
  "items": [
    {
      "instance_id": 1,
      "label": "Scissors",
      "box_id": "BOX-001",
      "track_ids": [3, 7],
      "first_frame": 12,
      "first_ts": 12.0,
      "status": "in_box",
      "exit_frame": null,
      "exit_ts": null
    }
  ],
  "by_box": { "BOX-001": [ ... ] },
  "summary": {
    "total_unique_items": 4,
    "items_in_box": 3,
    "items_removed": 1
  }
}
```

### `detection_log.csv`
One row per YOLO detection event. Columns: `frame_index`, `filename`, `timestamp`, `yolo_label`, `confidence`, `refined_label`, `track_id`, `instance_id`, `box_id`, `entry_detected`, `exit_detected`, `hand_detected`, `is_uncertain`, `plausibility_discarded`.

---

## References

- Jocher, G., & Qiu, J. (2024). Ultralytics YOLO11. https://github.com/ultralytics/ultralytics
- Zhang, Y. et al. (2022). ByteTrack. ECCV 2022. https://doi.org/10.1007/978-3-031-20047-2_1
- Meta. (2026). Llama Models. https://github.com/meta-llama/llama-models
- Sandler, M. et al. (2018). MobileNetV2. CVPR 2018.
