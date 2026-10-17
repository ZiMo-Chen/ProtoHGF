"""DVTOD HBB evaluator matching master/val_RGFNet.py.

The master script hard-codes one checkpoint. This version keeps the same
validation settings while accepting a checkpoint and output name so it can be
used for each ablation variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_rgfnet(
    checkpoint: str | Path,
    data: str | Path,
    *,
    project: str | Path,
    name: str,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    workers: int = 8,
    conf: float = 0.05,
    iou: float = 0.5,
) -> dict[str, float | str | int]:
    """Run the same HBB validation configuration used by master/val_RGFNet.py."""
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    metrics = YOLO(str(checkpoint)).val(
        data=str(Path(data).expanduser().resolve()),
        split="val",
        imgsz=imgsz,
        batch=batch,
        use_simotm="RGBRGB6C",
        channels=6,
        conf=conf,
        iou=iou,
        device=device,
        workers=workers,
        project=str(Path(project).expanduser().resolve()),
        name=name,
        plots=False,
        exist_ok=True,
    )
    return {
        "checkpoint": str(checkpoint),
        "data": str(Path(data).expanduser().resolve()),
        "project": str(Path(project).expanduser().resolve()),
        "name": name,
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
        "nms_iou": iou,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", default="runs/val_RGFNet")
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    result = evaluate_rgfnet(
        args.checkpoint, args.data, project=args.project, name=args.name,
        imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
        conf=args.conf, iou=args.iou,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
