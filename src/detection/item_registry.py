##############################################
# IMPROVED ITEM REGISTRY - Fixes label drift
#
# UPDATED: Added mark_removed() method for exit detection integration
##############################################

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
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
            "label":       self.refined_label,
            "box_id":      self.box_id,
            "track_ids":   sorted(list(self.track_ids)),
            "first_frame": self.first_frame,
            "last_frame":  self.last_frame,
            "first_ts":    self.first_ts,
            "last_ts":     self.last_ts,
            "status":      self.status,
            "exit_frame":  self.exit_frame,
            "exit_ts":     self.exit_ts,
        }


class ItemRegistry:

    def __init__(self,
                 same_item_window: int = 20,
                 label_similarity_threshold: float = 0.5):  # RAISED from 0.3

        self.same_item_window           = same_item_window
        self.label_similarity_threshold = label_similarity_threshold

        self._instances:        Dict[int, ItemInstance] = {}
        self._track_to_instance: Dict[int, int]         = {}
        self._next_id = 1

        # Category-based synonyms for semantic matching
        self._synonyms = {
            'bottle': {
                'bottle', 'water bottle', 'hand sanitizer', 'hand soap',
                'soap bottle', 'lotion bottle', 'shampoo bottle', 'sanitizer',
                'body wash', 'dish soap', 'cleaning spray', 'spray bottle'
            },
            'calculator': {
                'calculator', 'calc', 'adding machine',
                'phone', 'smartphone', 'cell phone', 'mobile phone', 'cellphone',
                'iphone', 'android', 'mobile', 'telephone',
                'remote', 'remote control'  # YOLO often confuses these
            },
            'book': {
                'book', 'notebook', 'journal', 'textbook', 'novel', 'diary',
                'planner', 'agenda', 'notepad'
            },
            'remote': {'remote', 'remote control', 'tv remote', 'controller'},
            'pen':    {'pen', 'ballpoint', 'marker', 'highlighter', 'sharpie'},
            'scissors': {'scissors', 'shears', 'snips'},
            'tape': {
                'tape', 'duct tape', 'masking tape', 'packing tape',
                'scotch tape', 'adhesive tape'
            },
            'charger': {
                'charger', 'phone charger', 'cable', 'charging cable',
                'power cord', 'usb cable', 'adapter'
            },
            'headphones': {
                'headphones', 'earbuds', 'earphones', 'airpods', 'headset'
            },
            'glasses': {
                'glasses', 'eyeglasses', 'sunglasses', 'spectacles',
                'reading glasses'
            },
            'watch':  {'watch', 'wristwatch', 'smartwatch', 'timepiece'},
            'wallet': {'wallet', 'purse', 'billfold', 'cardholder'},
            'tissue box': {
                'tissue box', 'tissue', 'kleenex', 'tissues',
                'tissue dispenser', 'facial tissue',
            },
            'screwdriver': {
                'screwdriver', 'screw driver', 'flathead', 'phillips',
                'phillips head', 'flathead screwdriver', 'tool',
            },
            'toilet paper': {
                'toilet paper', 'toilet roll', 'paper roll', 'roll',
                'tp', 'bathroom tissue',
            },
            'bag': {
                'bag', 'handbag', 'purse', 'suitcase', 'luggage',
                'backpack', 'duffel bag', 'tote bag', 'gym bag',
                'messenger bag', 'briefcase', 'carry-on', 'travel bag',
                'duffel', 'tote', 'shoulder bag', 'crossbody bag',
            },
        }

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def register_entry(self,
                       track_id:      int,
                       refined_label: str,
                       box_id:        str,
                       frame:         int,
                       timestamp:     float,
                       yolo_label:    str  = "",
                       is_uncertain:  bool = False) -> Tuple[int, bool]:

        yolo_lower = yolo_label.lower().strip()

        # Fast path - track already registered
        if track_id in self._track_to_instance:
            iid  = self._track_to_instance[track_id]
            inst = self._instances.get(iid)
            if inst and inst.status == "in_box":
                inst.update(track_id, frame, timestamp)
                return iid, False

        # IMPROVED: Try semantic matching on refined label first
        existing = self._find_matching_instance_semantic(
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
                    sim = self._semantic_similarity(
                        refined_label,
                        candidate.refined_label
                    )
                    if sim >= self.label_similarity_threshold:
                        existing = candidate

        if existing:
            existing.update(track_id, frame, timestamp)
            self._track_to_instance[track_id] = existing.instance_id

            logger.info(
                f"[REGISTRY] MERGED #{existing.instance_id}: '{refined_label}' "
                f"matched '{existing.refined_label}' "
                f"(sim={self._semantic_similarity(refined_label, existing.refined_label):.2f})"
            )

            return existing.instance_id, False

        # Create new instance
        iid  = self._next_id
        self._next_id += 1

        inst = ItemInstance(
            iid, refined_label, box_id,
            track_id, frame, timestamp,
            yolo_label=yolo_lower
        )

        self._instances[iid]            = inst
        self._track_to_instance[track_id] = iid

        logger.info(
            f"[REGISTRY] NEW #{iid}: '{refined_label}' "
            f"(yolo='{yolo_lower}') -> {box_id}"
        )

        return iid, True

    def mark_removed(self, instance_id: int, frame: int, timestamp: float) -> bool:
        """
        Mark a registered item as removed from its box.

        Called by ExitDetector once a removal has been confirmed (either by
        LLaMA visual verification or by the geometric absence fallback).

        Args:
            instance_id: The item's unique instance ID.
            frame:       Frame number at which removal was confirmed.
            timestamp:   Timestamp (seconds) of removal.

        Returns:
            True if the item was found and updated, False if not found.
        """
        inst = self._instances.get(instance_id)
        if inst is None:
            logger.warning(f"[REGISTRY] mark_removed: instance #{instance_id} not found")
            return False

        if inst.status == "removed":
            # Already marked — idempotent, no-op
            return True

        inst.status     = "removed"
        inst.exit_frame = frame
        inst.exit_ts    = timestamp

        logger.info(
            f"[REGISTRY] #{instance_id} '{inst.refined_label}' "
            f"marked REMOVED at frame={frame} t={timestamp:.2f}s"
        )
        return True

    # ==========================================================
    # EXPORT / ACCESS METHODS
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
                "items_in_box":       len(self.get_active_items()),
                "items_removed":      len(self.get_removed_items()),
                "unique_labels":      self.get_unique_labels(),
            }
        }

    # ==========================================================
    # IMPROVED MATCHING LOGIC - Semantic similarity
    # ==========================================================

    def _semantic_similarity(self, label1: str, label2: str) -> float:
        """
        Compute semantic similarity between labels using multiple strategies.

        Returns score between 0.0 and 1.0:
        - 1.0 = exact match
        - 0.85 = same category (e.g., both are bottles)
        - 0.7  = substring containment
        - 0.0-0.5 = token overlap or string similarity
        """
        l1 = label1.lower().strip()
        l2 = label2.lower().strip()

        # Exact match
        if l1 == l2:
            return 1.0

        # Category synonyms — HIGH priority match
        for category, words in self._synonyms.items():
            contains1 = any(w in l1 for w in words)
            contains2 = any(w in l2 for w in words)

            if contains1 and contains2:
                logger.debug(
                    f"[SIMILARITY] '{label1}' & '{label2}' both in '{category}' → 0.85"
                )
                return 0.85

        # Substring containment
        if l1 in l2 or l2 in l1:
            logger.debug(
                f"[SIMILARITY] '{label1}' substring of '{label2}' → 0.7"
            )
            return 0.7

        # Token overlap (Jaccard)
        tokens1 = set(l1.split())
        tokens2 = set(l2.split())
        if tokens1 and tokens2:
            intersection = tokens1 & tokens2
            union        = tokens1 | tokens2
            jaccard      = len(intersection) / len(union)
            if jaccard > 0:
                logger.debug(
                    f"[SIMILARITY] '{label1}' & '{label2}' token overlap → {jaccard:.2f}"
                )
                return jaccard

        # String similarity (difflib)
        ratio = SequenceMatcher(None, l1, l2).ratio()
        if ratio > 0.6:
            logger.debug(
                f"[SIMILARITY] '{label1}' & '{label2}' string match → {ratio*0.5:.2f}"
            )
            return ratio * 0.5

        return 0.0

    def _find_matching_instance_semantic(self,
                                          refined_label:  str,
                                          box_id:         str,
                                          current_frame:  int) -> Optional[ItemInstance]:
        """
        Find matching instance using SEMANTIC similarity instead of exact match.
        This is the key change that prevents label drift.
        """
        best       = None
        best_score = 0.0
        best_diff  = self.same_item_window + 1

        for inst in self._instances.values():
            if inst.status != "in_box":
                continue
            if inst.box_id != box_id:
                continue

            diff = current_frame - inst.last_frame
            if not (0 <= diff <= self.same_item_window):
                continue

            similarity = self._semantic_similarity(refined_label, inst.refined_label)

            if similarity >= self.label_similarity_threshold:
                if similarity > best_score or (similarity == best_score and diff < best_diff):
                    best_score = similarity
                    best_diff  = diff
                    best       = inst

        if best:
            logger.debug(
                f"[MATCH] '{refined_label}' → instance #{best.instance_id} "
                f"'{best.refined_label}' (score={best_score:.2f}, age={best_diff})"
            )

        return best

    def _find_same_yolo_origin(self,
                                yolo_lower:    str,
                                box_id:        str,
                                current_frame: int) -> Optional[ItemInstance]:
        """Find instance with same YOLO origin (fallback matcher)."""
        best      = None
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
                    best      = inst

        return best