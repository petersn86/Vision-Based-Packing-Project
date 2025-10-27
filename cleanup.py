##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'cleanup.py'
#
# Description:
# Safely deletes all frame image files in the
# '../data/frames/' directory, preserving the
# '.gitkeep' placeholder file.
##############################################

import os

def cleanup_frames(frames_dir: str = "data/frames"):

    if not os.path.exists(frames_dir):
        print(f"[ERROR] Directory not found: {frames_dir}")
        return

    deleted = 0
    for filename in os.listdir(frames_dir):
        file_path = os.path.join(frames_dir, filename)

        # Skip directories and .gitkeep
        if os.path.isdir(file_path) or filename == ".gitkeep":
            continue

        try:
            os.remove(file_path)
            deleted += 1
        except Exception as e:
            print(f"[WARNING] Could not delete {filename}: {e}")

    print(f"[INFO] Cleanup complete. Deleted {deleted} file(s).")


if __name__ == "__main__":
    cleanup_frames()
