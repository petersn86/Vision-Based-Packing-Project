##############################################
# SAFE + COMPLETE ITEM REGISTRY
##############################################

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ItemInstance:

    def __init__(self,
                 instance_id: int,
                 refined_label: str,
                 box_id: str,
                 track_id: int,
                 frame: int,
                 timestamp: float,
                 yolo_label: str = ""):

        self.instance_id   = instance_id
        self.label         = refined_label
        self.refined_label = refined_label
        self.yolo_label    = yolo_label.lower().strip()
        self.box_id        = box_id

        self.track_ids   = {track_id}
        self.first_frame = frame
        self.last_frame  = frame
        self.first_ts    = timestamp
        self.last_ts     = timestamp

        self.status      = "in_box"
        self.exit_frame  = None
        self.exit_ts     = None

    def update(self, track_id: int, frame: int, timestamp: float):
        self.track_ids.add(track_id)
        self.last_frame = frame
        self.last_ts    = timestamp

    def to_dict(self) -> Dict:
        return {
            "instance_id": self.instance_id,
            "label": self.refined_label,
            "box_id": self.box_id,
            "track_ids": sorted(list(self.track_ids)),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "status": self.status,
            "exit_frame": self.exit_frame,
            "exit_ts": self.exit_ts,
        }


class ItemRegistry:

    def __init__(self,
                 same_item_window: int = 20,
                 label_similarity_threshold: float = 0.3):

        self.same_item_window = same_item_window
        self.label_similarity_threshold = label_similarity_threshold

        self._instances: Dict[int, ItemInstance] = {}
        self._track_to_instance: Dict[int, int] = {}
        self._next_id = 1

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def register_entry(self,
                       track_id: int,
                       refined_label: str,
                       box_id: str,
                       frame: int,
                       timestamp: float,
                       yolo_label: str = "",
                       is_uncertain: bool = False) -> Tuple[int, bool]:

        yolo_lower = yolo_label.lower().strip()

        # Fast path
        if track_id in self._track_to_instance:
            iid = self._track_to_instance[track_id]
            inst = self._instances.get(iid)
            if inst and inst.status == "in_box":
                inst.update(track_id, frame, timestamp)
                return iid, False

        # Refined-label matching first
        existing = self._find_matching_instance(
            refined_label, box_id, frame
        )

        # Safe YOLO fallback
        if existing is None and yolo_lower:
            candidate = self._find_same_yolo_origin(
                yolo_lower, box_id, frame
            )

            if candidate:

                if is_uncertain:
                    existing = candidate
                else:
                    sim = self._token_similarity(
                        refined_label,
                        candidate.refined_label
                    )
                    if sim >= self.label_similarity_threshold:
                        existing = candidate

        if existing:
            existing.update(track_id, frame, timestamp)
            self._track_to_instance[track_id] = existing.instance_id
            return existing.instance_id, False

        # Create new
        iid = self._next_id
        self._next_id += 1

        inst = ItemInstance(
            iid, refined_label, box_id,
            track_id, frame, timestamp,
            yolo_label=yolo_lower
        )

        self._instances[iid] = inst
        self._track_to_instance[track_id] = iid

        logger.info(
            f"[REGISTRY] NEW #{iid}: '{refined_label}' "
            f"(yolo='{yolo_lower}') -> {box_id}"
        )

        return iid, True

    # ==========================================================
    # EXPORT / ACCESS METHODS (required by main.py)
    # ==========================================================

    def get_unique_labels(self) -> List[str]:
        labels = set()
        for inst in self._instances.values():
            if inst.status == "in_box":
                labels.add(inst.refined_label)
        return sorted(labels)

    def get_unique_labels_for_box(self, box_id: str) -> List[str]:
        labels = set()
        for inst in self._instances.values():
            if inst.status == "in_box" and inst.box_id == box_id:
                labels.add(inst.refined_label)
        return sorted(labels)

    def get_all_items(self) -> List[ItemInstance]:
        return list(self._instances.values())

    def get_active_items(self) -> List[ItemInstance]:
        return [i for i in self._instances.values()
                if i.status == "in_box"]

    def get_removed_items(self) -> List[ItemInstance]:
        return [i for i in self._instances.values()
                if i.status == "removed"]

    def export_to_dict(self) -> Dict:
        items_by_box = defaultdict(list)

        for inst in self._instances.values():
            items_by_box[inst.box_id].append(inst.to_dict())

        return {
            "items": [inst.to_dict() for inst in self._instances.values()],
            "by_box": dict(items_by_box),
            "summary": {
                "total_unique_items": len(self._instances),
                "items_in_box": len(self.get_active_items()),
                "items_removed": len(self.get_removed_items()),
                "unique_labels": self.get_unique_labels(),
            }
        }

    # ==========================================================
    # MATCHING LOGIC
    # ==========================================================

    def _find_same_yolo_origin(self,
                                yolo_lower: str,
                                box_id: str,
                                current_frame: int) -> Optional[ItemInstance]:

        best = None
        best_diff = self.same_item_window + 1

        for inst in self._instances.values():
            if inst.status != "in_box":
                continue
            if inst.box_id != box_id:
                continue
            if inst.yolo_label != yolo_lower:
                continue

            diff = current_frame - inst.last_frame
            if 0 <= diff <= self.same_item_window:
                if diff < best_diff:
                    best_diff = diff
                    best = inst

        return best

    def _find_matching_instance(self,
                                 refined_label: str,
                                 box_id: str,
                                 current_frame: int) -> Optional[ItemInstance]:

        label_lower = refined_label.lower().strip()

        best = None
        best_diff = self.same_item_window + 1

        for inst in self._instances.values():
            if inst.status != "in_box":
                continue
            if inst.box_id != box_id:
                continue

            diff = current_frame - inst.last_frame
            if not (0 <= diff <= self.same_item_window):
                continue

            if inst.refined_label.lower().strip() == label_lower:
                if diff < best_diff:
                    best_diff = diff
                    best = inst

        return best

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
