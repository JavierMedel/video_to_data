import json, sys
from pathlib import Path
import numpy as np
import torch

bbox_path, track_path, human_path, object_path = map(Path, sys.argv[1:5])
detections = json.loads(bbox_path.read_text())

def valid(b):
    try:
        a = np.array([b[k] for k in ("x0","y0","x1","y1")], dtype=float)
        return np.isfinite(a).all() and a[2] > a[0] and a[3] > a[1]
    except (KeyError, TypeError, ValueError):
        return False

if not human_path.exists():
    tracks = torch.load(track_path, map_location="cpu", weights_only=False)
    boxes = np.asarray(tracks["bbox_track"])
    candidates = [(i,b) for i,b in enumerate(boxes)
                  if len(b)==4 and np.isfinite(b).all()
                  and b[2]>b[0] and b[3]>b[1]]
    if not candidates:
        raise SystemExit("No valid human tracking box")
    i,b = candidates[0]
    human_path.write_text(json.dumps({"prompts":[{
        "frame_index": int(i), "object_id": 0,
        "box": dict(zip(("x0","y0","x1","y1"), map(float,b)))
    }]}, indent=2))
    print("Created human prompt at frame", i)

if not object_path.exists():
    items = [x for x in detections.get("0", []) if valid(x.get("box",{}))]
    if not items:
        raise SystemExit("No object detection in frame 0; manual prompt needed")
    x = max(items, key=lambda x: float(x.get("confidence",0)))
    object_path.write_text(json.dumps({"prompts":[{
        "frame_index": 0, "object_id": 1, "box": x["box"]
    }]}, indent=2))
    print("Created object prompt:", x.get("label"), x.get("confidence"))

for p in (human_path, object_path):
    if not p.is_file():
        raise SystemExit(f"Prompt file missing: {p}")
    print("VERIFIED", p)
