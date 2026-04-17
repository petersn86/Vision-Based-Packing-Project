##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'qr_detector.py'
#
# Description:
# QRDetector has been retired and superseded by BarcodeScanner
# (detection/barcode_scanner.py), which supports both 1D barcodes
# and QR codes with multi-scale preprocessing and frame persistence.
#
# BoxItemMapper is retained as a lightweight utility for maintaining
# track_id <-> box_id relationships outside of the ItemRegistry.
##############################################

from typing import List, Dict, Optional, Tuple


class BoxItemMapper:
    """
    Maintains mappings between boxes and items over time.

    Provides a simple lookup from track_id -> box_id and
    box_id -> list of (track_id, label, timestamp) tuples.
    Used as a utility alongside ItemRegistry for cross-referencing.
    """

    def __init__(self):
        self.box_items = {}   # box_id -> set of (track_id, label, timestamp)
        self.item_boxes = {}  # track_id -> (box_id, timestamp)

    def add_mapping(self,
                    track_id:  int,
                    label:     str,
                    box_id:    str,
                    timestamp: float):
        """
        Add item-to-box mapping.

        Args:
            track_id:  Object tracking ID
            label:     Item label
            box_id:    Box identifier
            timestamp: Detection timestamp
        """
        if box_id not in self.box_items:
            self.box_items[box_id] = set()

        self.box_items[box_id].add((track_id, label, timestamp))
        self.item_boxes[track_id] = (box_id, timestamp)

    def get_box_contents(self, box_id: str) -> List[Tuple]:
        """Get all items recorded in a box, sorted by timestamp."""
        return sorted(list(self.box_items.get(box_id, set())))

    def get_item_box(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Get the (box_id, timestamp) assignment for a tracked item."""
        return self.item_boxes.get(track_id)

    def get_all_boxes(self) -> Dict[str, List]:
        """Return all boxes with their item lists sorted by timestamp."""
        return {
            box_id: sorted(list(items), key=lambda x: x[2])
            for box_id, items in self.box_items.items()
        }

    def export_to_dict(self) -> Dict:
        """Export all mappings as a serialisable dictionary."""
        return {
            'boxes': {
                box_id: [
                    {
                        'track_id':  track_id,
                        'label':     label,
                        'timestamp': timestamp,
                    }
                    for track_id, label, timestamp in sorted(items, key=lambda x: x[2])
                ]
                for box_id, items in self.box_items.items()
            },
            'summary': {
                'total_boxes': len(self.box_items),
                'total_items': len(self.item_boxes),
            },
        }