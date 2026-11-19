"""AdamW training profile for the finest-resolution box head in a YOLO detector."""
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model


BOX_LR_MULTIPLIER = 1.5


class FetlockBoxTrainer(DetectionTrainer):
    def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        if name != "AdamW":
            raise ValueError("The fetlock box profile requires AdamW")
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        head = unwrap_model(model).model[-1]
        finest = int(head.stride.argmin().item())
        branches = [head.one2many["box_head"][finest]]
        if head.end2end:
            branches.append(head.one2one["box_head"][finest])
        box_ids = {id(param) for branch in branches for param in branch.parameters() if param.requires_grad}
        if not box_ids:
            raise ValueError("The finest-resolution box head has no trainable parameters")

        # Keep Ultralytics' weight decay, normalization and bias warmup rules in each new group.
        for group in list(optimizer.param_groups):
            selected = [param for param in group["params"] if id(param) in box_ids]
            if selected:
                group["params"] = [param for param in group["params"] if id(param) not in box_ids]
                optimizer.add_param_group({**group, "params": selected, "lr": lr * BOX_LR_MULTIPLIER})
        return optimizer
