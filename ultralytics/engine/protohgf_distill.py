"""Hook adapter that connects TM-Calib to this repository's dual-teacher trainer."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ultralytics.engine.tm_calib import MultiScaleTMCalibLoss


class ProtoHGFDistillation(nn.Module):
    """Collect teacher/student pyramid features and compute TM-Calib.

    The default layer indices match the paper's YOLOv8m student YAML and the
    standard YOLOv8m unimodal teachers in this repository.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model_rgb: nn.Module,
        teacher_model_thermal: nn.Module,
        channels: Sequence[int] = (192, 384, 576),
        student_rgb_layers: Sequence[int] = (11, 16, 21),
        student_thermal_layers: Sequence[int] = (12, 17, 22),
        teacher_layers: Sequence[int] = (4, 6, 8),
    ):
        super().__init__()
        student = self._unwrap(student_model)
        teacher_rgb = self._unwrap(teacher_model_rgb)
        teacher_thermal = self._unwrap(teacher_model_thermal)
        lengths = {
            len(channels),
            len(student_rgb_layers),
            len(student_thermal_layers),
            len(teacher_layers),
        }
        if len(lengths) != 1:
            raise ValueError("All channel and hook-layer lists must have the same length")

        self.loss = MultiScaleTMCalibLoss(channels=channels)
        self._teacher_rgb_modules = [teacher_rgb.model[index].cv2 for index in teacher_layers]
        self._teacher_thermal_modules = [teacher_thermal.model[index].cv2 for index in teacher_layers]
        self._student_rgb_modules = [student.model[index].cv2 for index in student_rgb_layers]
        self._student_thermal_modules = [student.model[index].cv2 for index in student_thermal_layers]
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.clear_feat()

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _capture(outputs: list[torch.Tensor]):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            outputs.append(output)

        return hook

    def register_hook(self) -> None:
        """Register feature hooks once at the beginning of an epoch."""
        self.remove_handle_()
        self.clear_feat()
        module_groups = (
            (self._teacher_rgb_modules, self.teacher_rgb_outputs),
            (self._teacher_thermal_modules, self.teacher_thermal_outputs),
            (self._student_rgb_modules, self.student_rgb_outputs),
            (self._student_thermal_modules, self.student_thermal_outputs),
        )
        for modules, outputs in module_groups:
            for module in modules:
                self._handles.append(module.register_forward_hook(self._capture(outputs)))

    def get_loss(self, _rgb_image: torch.Tensor | None = None) -> torch.Tensor:
        """Return the legacy three-slot loss vector expected by BaseTrainer."""
        total = self.loss(
            self.teacher_rgb_outputs,
            self.student_rgb_outputs,
            self.teacher_thermal_outputs,
            self.student_thermal_outputs,
        )
        zero = total.new_zeros(())
        return torch.stack((zero, total, zero))

    def get_feat(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.student_rgb_outputs, self.student_thermal_outputs

    def clear_feat(self) -> None:
        # Forward-hook closures retain the list objects passed to _capture().
        # Replacing those lists after a batch would make subsequent hook writes
        # invisible through these attributes, so preserve their identity and
        # clear them in place.
        for name in (
            "teacher_rgb_outputs",
            "teacher_thermal_outputs",
            "student_rgb_outputs",
            "student_thermal_outputs",
        ):
            outputs = getattr(self, name, None)
            if outputs is None:
                setattr(self, name, [])
            else:
                outputs.clear()

    def remove_handle_(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
