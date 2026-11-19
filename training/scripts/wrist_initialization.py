"""Transfer the pinned GRAZPEDWRI YOLO26s feature and box weights to a trusted model."""
import hashlib

import torch


REPOSITORY = "Crimson-Dawn/grazpedwri-yolo26-checkpoints"
REVISION = "01213afe2b4e978585339b3a0d5c6e23b38d538b"
SOURCE_FILE = ("runs/baseline_yolo26s_640_100epochs_seed42_from50_20260917_163013_311680Z/"
               "epoch_100/best.pt")
WRIST_SHA256 = "83589efd0ddc7929a3132289db359d03b107c70f5a26bc3da13b6a81b8b71971"
COCO_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"


class ModuleState:
    """Inert stand-in for serialized module and config objects. No third-party code is imported."""


def module_tensors(module, prefix=""):
    for name, tensor in module._parameters.items():
        if tensor is not None:
            yield prefix + name, tensor.detach()
    for name, tensor in module._buffers.items():
        if tensor is not None and name not in module._non_persistent_buffers_set:
            yield prefix + name, tensor.detach()
    for name, child in module._modules.items():
        if child is not None:
            yield from module_tensors(child, prefix + name + ".")


def initialize_from_wrist(target, path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != WRIST_SHA256:
        raise ValueError("腕片权重与固定的 Hugging Face SHA-256 不一致")
    names = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    for name in names:
        if not (name.startswith(("torch.nn.modules.", "ultralytics.nn.modules."))
                or name in ("ultralytics.nn.tasks.DetectionModel", "ultralytics.utils.IterableSimpleNamespace")):
            raise ValueError(f"未知的权重对象 {name}")
    with torch.serialization.safe_globals([(ModuleState, name) for name in names]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source = checkpoint["ema"]
    if source.names != {0: "fracture"} or source.yaml["yaml_file"] != "yolo26s.yaml":
        raise ValueError("腕片权重的数据类别或结构与预期不符")
    original = target.state_dict()
    head = f"model.{len(target.model) - 1}."
    class_branches = (head + "cv3.", head + "one2one_cv3.")
    retained = {key for key in original if key.startswith(class_branches)}
    transfer = {key: value for key, value in module_tensors(source) if not key.startswith(class_branches)}
    if not retained or transfer.keys() != original.keys() - retained:
        raise ValueError("腕片和目标模型的非分类层名称不匹配")
    if any(value.shape != original[key].shape for key, value in transfer.items()):
        raise ValueError("腕片和目标模型的非分类层形状不匹配")
    target.load_state_dict(transfer, strict=False)
    return {"repository": REPOSITORY, "revision": REVISION, "file": SOURCE_FILE, "sha256": WRIST_SHA256,
            "checkpoint_epoch_index": checkpoint["epoch"], "source_label": source.names[0],
            "transferred_tensors": len(transfer), "class_branch_tensors_left_at_baseline": len(retained)}
