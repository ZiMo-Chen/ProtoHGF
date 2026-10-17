"""Explicit YOLOv8m initialization for the dual-branch ProtoHGF detector."""

from __future__ import annotations

from torch import nn


# Standard YOLOv8m backbone -> the two modality branches. Fusion modules have
# no single-modality counterpart and retain their normal random initialization.
BRANCH_LAYERS = {
    0: (3, 4),
    1: (5, 6),
    2: (7, 8),
    3: (9, 10),
    4: (11, 12),
    5: (14, 15),
    6: (16, 17),
    7: (19, 20),
    8: (21, 22),
}

# Standard YOLOv8m neck/head -> the unchanged ProtoHGF neck/head.
SHARED_LAYERS = {
    9: 24,
    12: 27,
    15: 30,
    16: 31,
    18: 33,
    19: 34,
    21: 36,
    22: 37,
}


def initialize_protohgf_from_yolov8m(student: nn.Module, source: nn.Module) -> tuple[int, int]:
    """Copy matching YOLOv8m weights into both student branches and shared head.

    Returns:
        (number of copied parameter elements, total student parameter elements).

    Raises:
        ValueError: The model layers do not match the known YOLOv8m/ProtoHGF layout.
    """
    if len(source.model) != 23 or len(student.model) != 38:
        raise ValueError("Expected a 23-layer YOLOv8m source and a 38-layer ProtoHGF detection student")

    source_state = source.float().state_dict()
    target_state = student.state_dict()
    transferred = {}
    layer_pairs = [(old, new) for old, news in BRANCH_LAYERS.items() for new in news]
    layer_pairs.extend(SHARED_LAYERS.items())
    for old, new in layer_pairs:
        if type(source.model[old]) is not type(student.model[new]):
            raise ValueError(f"Layer type mismatch at YOLOv8m {old} -> ProtoHGF {new}")
        source_prefix = f"model.{old}."
        target_prefix = f"model.{new}."
        layer_items = [(key, value) for key, value in source_state.items() if key.startswith(source_prefix)]
        copied = 0
        for key, value in layer_items:
            target_key = target_prefix + key[len(source_prefix):]
            if target_key in target_state and target_state[target_key].shape == value.shape:
                transferred[target_key] = value
                copied += 1
        # The class prediction layer may differ when the source has COCO's 80
        # classes and the DVTOD student has three classes. All other layers
        # must copy completely, or the architecture has changed unexpectedly.
        if not layer_items or (old != 22 and copied != len(layer_items)):
            raise ValueError(f"Incomplete weight transfer at YOLOv8m {old} -> ProtoHGF {new}: "
                             f"{copied}/{len(layer_items)} state entries")

    student.load_state_dict(transferred, strict=False)
    student_params = dict(student.named_parameters())
    copied_elements = sum(student_params[key].numel() for key in transferred if key in student_params)
    total_elements = sum(parameter.numel() for parameter in student_params.values())
    return copied_elements, total_elements
