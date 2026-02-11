##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'item_registry.py'
#
# Description:
# Provides stable item instance IDs that survive track ID
# fragmentation. Deduplicates detections by (label + box_id +
# recency) rather than by track ID.
#
# Why this exists:
#   ByteTrack assigns a new track ID whenever it loses and
#   re-finds an object. For a held item being placed into a box,
#   this can happen every single frame. items_entered_boxes[track_id]
#   therefore fails to deduplicate — each new ID passes the guard
#   and the same physical item gets recorded multiple times.
#
#   ItemRegistry fixes this by asking a different question:
#   "Have we already seen THIS LABEL go into THIS BOX recently?"
#   If yes → same item, don't record again, just update the mapping.
#   If no  → new item, assign a fresh instance_id and record it.
#
# Future use:
#   register_exit() is already implemented and ready for the exit
#   detection module. Exit detection calls it with the track_id of
#   an object that has left the box region; the registry resolves
#   the stable instance_id and marks it 'removed'.
#
##############################################

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ItemInstance:
    """Represents one unique physical item detected in the video."""

    def __init__(self,
                 instance_id: int,
                 label: str,
                 box_id: str,
                 track_id: int,
                 frame: int,
                 timestamp: float):
        self.instance_id    = instance_id
        self.label          = label
        self.box_id         = box_id
        self.track_ids      = {track_id}        # All track IDs seen for this item
        self.first_frame    = frame
        self.last_frame     = frame
        self.first_ts       = timestamp
        self.last_ts        = timestamp
        self.status         = 'in_box'          # 'in_box' | 'removed'
        self.exit_frame     = None
        self.exit_ts        = None
        self.refined_label  = None              # Set after LLaMA refinement

    def update(self, track_id: int, frame: int, timestamp: float):
        """Record a new sighting of this item."""
        self.track_ids.add(track_id)
        self.last_frame = frame
        self.last_ts    = timestamp

    def to_dict(self) -> Dict:
        return {
            'instance_id':  self.instance_id,
            'label':        self.refined_label or self.label,
            'raw_label':    self.label,
            'box_id':       self.box_id,
            'track_ids':    sorted(list(self.track_ids)),
            'first_frame':  self.first_frame,
            'last_frame':   self.last_frame,
            'first_ts':     self.first_ts,
            'last_ts':      self.last_ts,
            'status':       self.status,
            'exit_frame':   self.exit_frame,
            'exit_ts':      self.exit_ts,
        }


class ItemRegistry:
    """
    Maintains a deduplicated registry of items detected in boxes.

    Core logic:
        register_entry(track_id, label, box_id, frame, timestamp)
            → Returns (instance_id, is_new_item)

        If a matching item (same label, same box, seen within
        same_item_window frames) already exists with status='in_box',
        this is treated as a re-detection of the same physical item.
        The existing instance is updated and is_new_item=False.

        Otherwise a new ItemInstance is created and is_new_item=True.
        Only new items should trigger LLaMA refinement and be written
        to the final item list.

    Exit detection (future):
        register_exit(track_id, frame, timestamp)
            → Marks the item associated with track_id as 'removed'
    """

    def __init__(self, same_item_window: int = 60):
        """
        Args:
            same_item_window: Number of frames within which a re-detection
                              of the same label+box is considered the same
                              physical item rather than a new one.
                              At 0.5s frame interval, 60 frames = 30 seconds.
                              Increase if items are out of frame for longer.
        """
        self.same_item_window   = same_item_window
        self._instances: Dict[int, ItemInstance] = {}   # instance_id -> ItemInstance
        self._track_to_instance: Dict[int, int]  = {}   # track_id    -> instance_id
        self._next_id = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_entry(self,
                       track_id: int,
                       label: str,
                       box_id: str,
                       frame: int,
                       timestamp: float) -> Tuple[int, bool]:
        """
        Register an item entry event.

        Args:
            track_id:  Current tracker ID (may be volatile)
            label:     YOLO label (pre-LLaMA)
            box_id:    Box identifier string
            frame:     Current frame index
            timestamp: Current timestamp in seconds

        Returns:
            (instance_id, is_new_item)
            is_new_item=True  → first time we've seen this item, process it
            is_new_item=False → re-detection of existing item, skip LLaMA
        """
        # Fast path: this track_id is already mapped to an instance
        if track_id in self._track_to_instance:
            iid = self._track_to_instance[track_id]
            instance = self._instances[iid]
            if instance.status == 'in_box':
                instance.update(track_id, frame, timestamp)
                logger.debug(
                    f"[REGISTRY] Track#{track_id} → existing instance #{iid} "
                    f"'{instance.label}' (fast path)"
                )
                return iid, False

        # Slow path: search for a matching active instance
        existing = self._find_matching_instance(label, box_id, frame)
        if existing is not None:
            existing.update(track_id, frame, timestamp)
            self._track_to_instance[track_id] = existing.instance_id
            logger.debug(
                f"[REGISTRY] Track#{track_id} '{label}' → merged into "
                f"existing instance #{existing.instance_id} "
                f"(last seen frame {existing.last_frame}, "
                f"window={self.same_item_window})"
            )
            return existing.instance_id, False

        # Genuinely new item
        iid = self._next_id
        self._next_id += 1
        instance = ItemInstance(iid, label, box_id, track_id, frame, timestamp)
        self._instances[iid] = instance
        self._track_to_instance[track_id] = iid
        logger.info(
            f"[REGISTRY] NEW item #{iid}: Track#{track_id} '{label}' → {box_id} "
            f"at frame {frame} ({timestamp}s)"
        )
        return iid, True

    def set_refined_label(self, instance_id: int, refined_label: str):
        """Store the LLaMA-refined label for an instance."""
        if instance_id in self._instances:
            self._instances[instance_id].refined_label = refined_label

    def register_exit(self,
                      track_id: int,
                      frame: int,
                      timestamp: float) -> Optional[int]:
        """
        Mark the item associated with track_id as removed from its box.

        Called by exit detection when an item's bbox leaves the box region.

        Args:
            track_id:  Track ID of the departing object
            frame:     Frame index when exit was detected
            timestamp: Timestamp when exit was detected

        Returns:
            instance_id if an item was marked as removed, None if unknown track
        """
        iid = self._track_to_instance.get(track_id)
        if iid is None:
            logger.debug(f"[REGISTRY] register_exit: Track#{track_id} not in registry")
            return None

        instance = self._instances[iid]
        if instance.status == 'in_box':
            instance.status     = 'removed'
            instance.exit_frame = frame
            instance.exit_ts    = timestamp
            logger.info(
                f"[REGISTRY] EXIT #{iid}: '{instance.refined_label or instance.label}' "
                f"left {instance.box_id} at frame {frame} ({timestamp}s)"
            )
            return iid

        return None

    def get_instance(self, instance_id: int) -> Optional[ItemInstance]:
        return self._instances.get(instance_id)

    def get_instance_for_track(self, track_id: int) -> Optional[ItemInstance]:
        iid = self._track_to_instance.get(track_id)
        return self._instances.get(iid) if iid is not None else None

    def get_all_items(self) -> List[ItemInstance]:
        """Return all registered item instances."""
        return list(self._instances.values())

    def get_active_items(self) -> List[ItemInstance]:
        """Return items currently in a box (status='in_box')."""
        return [i for i in self._instances.values() if i.status == 'in_box']

    def get_removed_items(self) -> List[ItemInstance]:
        """Return items that have been removed from a box."""
        return [i for i in self._instances.values() if i.status == 'removed']

    def get_unique_labels(self) -> List[str]:
        """Sorted list of unique refined/raw labels across all instances."""
        labels = set()
        for inst in self._instances.values():
            labels.add(inst.refined_label or inst.label)
        return sorted(labels)

    def export_to_dict(self) -> Dict:
        """Full export for JSON serialisation."""
        items_by_box = defaultdict(list)
        for inst in self._instances.values():
            items_by_box[inst.box_id].append(inst.to_dict())

        return {
            'items':   [inst.to_dict() for inst in self._instances.values()],
            'by_box':  dict(items_by_box),
            'summary': {
                'total_unique_items': len(self._instances),
                'items_in_box':       len(self.get_active_items()),
                'items_removed':      len(self.get_removed_items()),
                'unique_labels':      self.get_unique_labels(),
            }
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_matching_instance(self,
                                 label: str,
                                 box_id: str,
                                 current_frame: int) -> Optional[ItemInstance]:
        """
        Find an existing active instance with the same label and box_id
        that was seen within same_item_window frames of current_frame.

        Uses case-insensitive label matching so "bottle" and "Bottle"
        resolve to the same item.
        """
        label_lower = label.lower().strip()
        best: Optional[ItemInstance] = None
        best_frame_diff = self.same_item_window + 1

        for inst in self._instances.values():
            if inst.status != 'in_box':
                continue
            if inst.box_id != box_id:
                continue
            if inst.label.lower().strip() != label_lower:
                # Also check refined label in case LLaMA changed it
                if (inst.refined_label or '').lower().strip() != label_lower:
                    continue

            frame_diff = current_frame - inst.last_frame
            if 0 <= frame_diff <= self.same_item_window:
                # Pick the most recently seen match
                if frame_diff < best_frame_diff:
                    best_frame_diff = frame_diff
                    best = inst

        return best
