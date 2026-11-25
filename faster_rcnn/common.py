"""Shared pieces for the Faster R-CNN pipeline: COCO dataset, model builder, checkpoints."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms import v2 as T

MODELS = {
    "resnet50_fpn_v2": (fasterrcnn_resnet50_fpn_v2, 5),
    "resnet50_fpn": (fasterrcnn_resnet50_fpn, 5),
    "mobilenet_v3_fpn": (fasterrcnn_mobilenet_v3_large_fpn, 6),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- dataset


def load_categories(coco_json):
    """Categories sorted by id. Label 0 is background, so category i gets label i + 1."""
    with open(coco_json, encoding="utf-8") as f:
        cats = json.load(f)["categories"]
    return sorted(cats, key=lambda c: c["id"])


class CocoDetection(torch.utils.data.Dataset):
    """Reads a COCO detection json (TrashCan instance/material, or any split made from it).

    `categories` fixes the label mapping so train/val/test all use the same one.
    `mosaic` is the probability of replacing an image by a 4-frame mosaic (training only).
    """

    def __init__(self, coco_json, image_dir, categories, transforms=None, mosaic=0.0):
        with open(coco_json, encoding="utf-8") as f:
            data = json.load(f)
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.mosaic = mosaic
        self.cat_to_label = {c["id"]: i + 1 for i, c in enumerate(categories)}
        self.images = sorted(data["images"], key=lambda im: im["id"])
        self.anns = {im["id"]: [] for im in self.images}
        for a in data["annotations"]:
            if a.get("iscrowd", 0) or a["image_id"] not in self.anns:
                continue
            self.anns[a["image_id"]].append(a)

    def __len__(self):
        return len(self.images)

    def _load(self, idx):
        """uint8 image [3, H, W], xyxy boxes [N, 4] and labels [N] of one image."""
        info = self.images[idx]
        img = read_image(str(self.image_dir / info["file_name"]), ImageReadMode.RGB)
        h, w = img.shape[-2:]
        boxes, labels = [], []
        for a in self.anns[info["id"]]:
            x, y, bw, bh = a["bbox"]
            # About 27% of TrashCan boxes run past the frame edge. Clip them here, because
            # SanitizeBoundingBoxes silently drops any box that leaves the image.
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + bw, w), min(y + bh, h)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_to_label[a["category_id"]])
        return (img, torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                torch.tensor(labels, dtype=torch.int64))

    def _mosaic(self, idx):
        """YOLO-style mosaic: this frame and three random others on a 2x2 canvas, cropped back to
        this frame's size around a random centre, so objects appear next to other videos' scenes."""
        img0, boxes0, labels0 = self._load(idx)
        H, W = img0.shape[-2:]
        canvas = torch.full((3, 2 * H, 2 * W), 114, dtype=torch.uint8)
        all_boxes, all_labels = [], []
        for k, j in enumerate([idx] + random.choices(range(len(self.images)), k=3)):
            img, boxes, labels = (img0, boxes0, labels0) if k == 0 else self._load(j)
            h, w = img.shape[-2:]
            s = min(H / h, W / w, 1.0)
            if s < 1:
                h, w = round(h * s), round(w * s)
                img = T.functional.resize(img, [h, w], antialias=True)
                boxes = boxes * s
            # Each frame touches the canvas centre: top-left, top-right, bottom-left, bottom-right.
            y0 = H - h if k < 2 else H
            x0 = W - w if k % 2 == 0 else W
            canvas[:, y0:y0 + h, x0:x0 + w] = img
            all_boxes.append(boxes + torch.tensor([x0, y0, x0, y0]))
            all_labels.append(labels)
        top = random.randint(0, H)
        left = random.randint(0, W)
        boxes = torch.cat(all_boxes) - torch.tensor([left, top, left, top])
        clipped = boxes.clone()
        clipped[:, 0::2] = clipped[:, 0::2].clamp(0, W)
        clipped[:, 1::2] = clipped[:, 1::2].clamp(0, H)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        cw, ch = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
        # Ultralytics' rule: drop boxes cut to under 10% of their area or under 2 px.
        keep = (cw > 2) & (ch > 2) & (cw * ch > 0.1 * area)
        return canvas[:, top:top + H, left:left + W], clipped[keep], torch.cat(all_labels)[keep]

    def __getitem__(self, idx):
        if self.mosaic and random.random() < self.mosaic:
            img, boxes, labels = self._mosaic(idx)
        else:
            img, boxes, labels = self._load(idx)
        h, w = img.shape[-2:]
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
            "labels": labels,
            "image_id": self.images[idx]["id"],
        }
        img = tv_tensors.Image(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def take_subset(dataset, coco_gt, n):
    """Keep only the first n images of a dataset (and its pycocotools ground truth), for quick tests."""
    dataset.images = dataset.images[:n]
    if coco_gt is not None:
        keep = {im["id"] for im in dataset.images}
        coco_gt.dataset["images"] = dataset.images
        coco_gt.dataset["annotations"] = [a for a in coco_gt.dataset["annotations"] if a["image_id"] in keep]
        coco_gt.createIndex()


def get_transforms(train, aug="basic"):
    t = []
    if train and aug != "none":
        t.append(T.RandomHorizontalFlip(0.5))
        if aug == "strong":
            # Underwater colour/brightness varies a lot between dives.
            t.append(T.RandomPhotometricDistort(p=0.5))
    t += [T.ToDtype(torch.float32, scale=True), T.SanitizeBoundingBoxes()]
    return T.Compose(t)


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------- model


def freeze_batchnorm(module):
    """Swap BatchNorm2d for FrozenBatchNorm2d, keeping the pretrained statistics.

    With batch size 4 the BN statistics are too noisy to update, so the
    backbone keeps the ones learned on COCO / ImageNet.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            frozen.weight.copy_(child.weight.detach())
            frozen.bias.copy_(child.bias.detach())
            frozen.running_mean.copy_(child.running_mean)
            frozen.running_var.copy_(child.running_var)
            setattr(module, name, frozen)
        else:
            freeze_batchnorm(child)
    return module


def build_model(name, num_classes, pretrained="coco", trainable_layers=3, min_size=(800,), max_size=1333):
    """Faster R-CNN with a new box predictor for `num_classes` (+ background).

    pretrained: "coco"     -> whole detector pretrained on COCO (transfer learning)
                "imagenet" -> only the backbone pretrained on ImageNet
                "none"     -> random init (also used to rebuild a model before loading a checkpoint)
    """
    builder, max_layers = MODELS[name]
    kwargs = dict(min_size=tuple(min_size), max_size=max_size)
    if pretrained == "coco":
        kwargs.update(weights="DEFAULT", trainable_backbone_layers=trainable_layers)
    elif pretrained == "imagenet":
        kwargs.update(weights=None, weights_backbone="DEFAULT", trainable_backbone_layers=trainable_layers)
    elif pretrained == "none":
        kwargs.update(weights=None, weights_backbone=None)
    else:
        raise ValueError(f"unknown pretrained mode {pretrained!r}")
    if not 0 <= trainable_layers <= max_layers:
        raise ValueError(f"{name} supports trainable_layers 0..{max_layers}")

    model = builder(**kwargs)
    if name == "resnet50_fpn_v2" and pretrained != "none":
        # v2 builds its ResNet with plain BatchNorm2d; v1/mobilenet already freeze it.
        freeze_batchnorm(model.backbone.body)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def save_checkpoint(path, model, args, categories, epoch, best_map, optimizer=None, scheduler=None, scaler=None):
    ckpt = {
        "model": model.state_dict(),
        "model_name": args.model,
        "categories": categories,
        "min_size": list(args.min_size),
        "max_size": args.max_size,
        "epoch": epoch,
        "best_map": best_map,
        "args": vars(args),
    }
    if optimizer is not None:
        ckpt.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
    torch.save(ckpt, path)


def load_model_from_checkpoint(path, device):
    """Rebuild the network from a checkpoint alone (no pretrained download needed)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cats = ckpt["categories"]
    model = build_model(ckpt["model_name"], len(cats), pretrained="none",
                        min_size=ckpt["min_size"], max_size=ckpt["max_size"])
    # A random-init rebuild uses BatchNorm2d where training used FrozenBatchNorm2d.
    # They are identical in eval mode; the only key difference is num_batches_tracked.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    bad = [k for k in missing + unexpected if not k.endswith("num_batches_tracked")]
    if bad:
        raise RuntimeError(f"checkpoint does not match model: {bad[:5]}")
    model.to(device).eval()
    return model, cats, ckpt