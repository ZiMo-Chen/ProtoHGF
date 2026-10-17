"""Train the paper configuration of ProtoHGF-Net.

Examples:
    python train_protohgf.py --dataset dvtod --rgb-teacher runs/.../best.pt \
        --thermal-teacher runs/.../best.pt

    python train_protohgf.py --dataset dronevehicle --rgb-teacher runs/.../best.pt \
        --thermal-teacher runs/.../best.pt --device 0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    data: str
    model: str
    epochs: int
    batch: int
    task: str


EXPERIMENTS = {
    "dronevehicle": Experiment(
        data="ultralytics/cfg/datasets/DroneVehicle.yaml",
        model="ultralytics/cfg/models/v8-RGBT/protohgf-yolov8m-obb.yaml",
        epochs=60,
        batch=8,
        task="obb",
    ),
    "dvtod": Experiment(
        data="ultralytics/cfg/datasets/DVTOD.yaml",
        model="ultralytics/cfg/models/v8-RGBT/protohgf-yolov8m.yaml",
        epochs=150,
        batch=16,
        task="detect",
    ),
    "flir": Experiment(
        data="ultralytics/cfg/datasets/FLIR_aligned.yaml",
        model="ultralytics/cfg/models/v8-RGBT/protohgf-yolov8m.yaml",
        epochs=36,
        batch=16,
        task="detect",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ProtoHGF-Net with TM-Calib")
    parser.add_argument("--dataset", choices=EXPERIMENTS, required=True)
    parser.add_argument("--rgb-teacher", help="RGB-only YOLOv8m best.pt (required with TM-Calib)")
    parser.add_argument("--thermal-teacher", help="Thermal-only YOLOv8m best.pt (required with TM-Calib)")
    parser.add_argument("--data", help="Override the dataset YAML from the selected profile")
    parser.add_argument("--model", help="Override the model YAML from the selected profile")
    parser.add_argument("--epochs", type=int, help="Override the paper's epoch count")
    parser.add_argument("--batch", type=int, help="Override the paper's batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--device", default="", help="CUDA device, e.g. 0 or 0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/ProtoHGF-Net")
    parser.add_argument("--name", help="Run name; defaults to the dataset profile name")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--resume", help="Resume from a previous last.pt checkpoint")
    parser.add_argument("--student-pretrained", help="YOLOv8m .pt to map into both student branches")
    parser.add_argument("--no-tmcalib", action="store_true", help="Disable TM-Calib and both teachers")
    args = parser.parse_args()
    if not args.no_tmcalib and (not args.rgb_teacher or not args.thermal_teacher):
        parser.error("--rgb-teacher and --thermal-teacher are required unless --no-tmcalib is set")
    if args.student_pretrained:
        if args.resume:
            parser.error("--student-pretrained cannot be combined with --resume")
        if EXPERIMENTS[args.dataset].task != "detect":
            parser.error("--student-pretrained currently supports the HBB detection model only")
        student_checkpoint = Path(args.student_pretrained).expanduser().resolve()
        if not student_checkpoint.is_file():
            parser.error(f"Student pretrained checkpoint does not exist: {student_checkpoint}")
        args.student_pretrained = str(student_checkpoint)
    return args


def main() -> None:
    args = parse_args()

    # PyTorch queues CUDA capability checks when it is imported. Restricting
    # CUDA_VISIBLE_DEVICES after another call has already enumerated all GPUs
    # can make those queued checks use a stale device count (PyTorch 2.3),
    # resulting in a DeferredCudaCallError. Apply the CLI selection before any
    # Ultralytics import, since importing Ultralytics imports PyTorch as well.
    requested_device = str(args.device).strip()
    if requested_device and requested_device.lower() not in {"cpu", "mps"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_device

    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.obb import OBBTrainer
    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.utils import DEFAULT_CFG

    experiment = EXPERIMENTS[args.dataset]
    overrides = {
        "model": args.model or experiment.model,
        "data": args.data or experiment.data,
        "pretrained": args.student_pretrained or False,
        "imgsz": args.imgsz,
        "epochs": args.epochs or experiment.epochs,
        "batch": args.batch or experiment.batch,
        "device": args.device,
        "workers": args.workers,
        # Legacy master/train_Dist.py inherits optimizer=auto from default.yaml.
        "optimizer": "auto",
        "lr0": 0.001,
        "momentum": 0.937,
        "amp": args.amp,
        "close_mosaic": 10,
        "project": args.project,
        "name": args.name or args.dataset,
    }
    if not args.no_tmcalib:
        rgb_teacher, _ = attempt_load_one_weight(args.rgb_teacher)
        thermal_teacher, _ = attempt_load_one_weight(args.thermal_teacher)
        for teacher in (rgb_teacher, thermal_teacher):
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        overrides.update(
            Distillation="MultiDistillation",
            Teacher_Model_RGB=rgb_teacher,
            Teacher_Model_IR=thermal_teacher,
            loss_type="TMCalib",
            distill_weight=1.0,
            online=False,
        )
    if args.resume:
        overrides["resume"] = args.resume

    DEFAULT_CFG.use_simotm = "RGBRGB6C"
    DEFAULT_CFG.channels = 6
    trainer_class = OBBTrainer if experiment.task == "obb" else DetectionTrainer
    if args.student_pretrained:
        from ultralytics.nn.protohgf_init import initialize_protohgf_from_yolov8m
        from ultralytics.utils import LOGGER

        class InitializedDetectionTrainer(DetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                student = super().get_model(cfg=cfg, weights=None, verbose=verbose)
                if weights is None:
                    raise ValueError("Student pretrained checkpoint was not loaded")
                copied, total = initialize_protohgf_from_yolov8m(student, weights)
                LOGGER.info(f"Initialized ProtoHGF student from {args.student_pretrained}: "
                            f"{copied:,}/{total:,} parameter elements copied")
                return student

        trainer_class = InitializedDetectionTrainer
    trainer_class(overrides=overrides).train()


if __name__ == "__main__":
    main()
