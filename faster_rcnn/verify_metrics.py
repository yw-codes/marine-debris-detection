"""Check metrics.py by scoring the ground truth against itself: mAP, P and R should be about 1.0.

Two passes over one split:
    raw json     annotations copied straight into detections (score 1.0)
    via dataset  the boxes and labels CocoDetection gives to training, mapped back the way
                 run_inference does it. Clipped boxes differ a little from the json, so this
                 pass can sit slightly below 1.0. It also shows how many boxes survive.

    python verify_metrics.py --json D:/debris/data/instances_val.json --images D:/debris/data/val
"""

import argparse

import torch

from common import CocoDetection, get_transforms, load_categories
from metrics import coco_evaluate, load_coco, precision_recall


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", required=True, help="ground-truth COCO json of one split")
    p.add_argument("--images", required=True)
    return p.parse_args()


def report(name, coco_gt, dets, categories):
    stats, _ = coco_evaluate(coco_gt, dets)
    pr = precision_recall(coco_gt, dets, categories)["best_f1"]
    print(f"{name:<12} {len(dets)} dets | mAP@.5:.95 {stats['mAP50_95']:.4f} | mAP@.5 {stats['mAP50']:.4f} | "
          f"P {pr['precision']:.4f} | R {pr['recall']:.4f}")
    return min(stats["mAP50_95"], stats["mAP50"], pr["precision"], pr["recall"])


def main():
    args = parse_args()
    categories = load_categories(args.json)
    coco_gt = load_coco(args.json)
    gt_anns = [a for a in coco_gt.dataset["annotations"] if not a.get("iscrowd", 0)]

    raw = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
           for a in gt_anns]
    worst = report("raw json", coco_gt, raw, categories)

    label_to_cat = {i + 1: c["id"] for i, c in enumerate(categories)}
    dataset = CocoDetection(args.json, args.images, categories, get_transforms(False))
    dets = []
    for k in range(len(dataset)):
        target = dataset[k][1]
        boxes = target["boxes"].as_subclass(torch.Tensor).clone()
        boxes[:, 2:] -= boxes[:, :2]
        for b, l in zip(boxes.tolist(), target["labels"].tolist()):
            dets.append({"image_id": target["image_id"], "category_id": label_to_cat[l], "bbox": b, "score": 1.0})
    report("via dataset", coco_gt, dets, categories)
    print(f"boxes kept for training: {len(dets)} / {len(gt_anns)}")

    print("PASS" if worst > 0.99 else "FAIL: metrics.py does not score perfect detections as perfect")


if __name__ == "__main__":
    main()