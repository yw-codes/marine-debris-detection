"""Evaluate a trained checkpoint on a COCO-format split (normally the shared test split).

Writes to --out:
    metrics.json      COCO mAP@[.5:.95] / mAP@.5 / size breakdown, P/R, inference speed
    per_class_ap.csv  AP per category (+ number of GT boxes)
    predictions.json  COCO results format, for the team's unified evaluation against YOLO
    pr_curve.png      precision-recall curves at IoU 0.5
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from common import (CocoDetection, collate_fn, get_transforms, load_categories, load_model_from_checkpoint,
                    pick_device, take_subset)
from metrics import (benchmark, coco_evaluate, load_coco, per_class_ap, plot_pr_curves, precision_recall,
                     run_inference)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="best.pt / last.pt from train.py")
    p.add_argument("--json", required=True, help="ground-truth COCO json of the split to evaluate")
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="fp16 inference on CUDA (also used for the speed test)")
    p.add_argument("--bench-images", type=int, default=200, help="images for the speed test, 0 to skip")
    p.add_argument("--conf", type=float, default=0.5, help="fixed confidence threshold for the extra P/R numbers")
    p.add_argument("--score-thresh", type=float, default=None,
                   help="drop detections below this score (torchvision default 0.05, Ultralytics val uses 0.001)")
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, categories, ckpt = load_model_from_checkpoint(args.weights, device)
    if args.score_thresh is not None:
        model.roi_heads.score_thresh = args.score_thresh
    if [c["id"] for c in load_categories(args.json)] != [c["id"] for c in categories]:
        raise SystemExit("this json has different categories from the ones the model was trained on")
    label_to_cat = {i + 1: c["id"] for i, c in enumerate(categories)}

    dataset = CocoDetection(args.json, args.images, categories, get_transforms(False))
    coco_gt = load_coco(args.json)
    if args.subset:
        take_subset(dataset, coco_gt, args.subset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                         collate_fn=collate_fn, pin_memory=device.type == "cuda")

    print(f"evaluating {args.weights} (epoch {ckpt['epoch']}) on {len(dataset)} images, device={device}")
    dets = run_inference(model, loader, device, label_to_cat, args.amp)
    (out / "predictions.json").write_text(json.dumps(dets))
    stats, E = coco_evaluate(coco_gt, dets, verbose=True)
    pr = precision_recall(coco_gt, dets, categories, fixed_conf=args.conf)
    rows = per_class_ap(E, coco_gt, categories)
    plot_pr_curves(E, categories, out / "pr_curve.png")

    speed = None
    if args.bench_images:
        speed = benchmark(model, dataset, device, n=args.bench_images, amp=args.amp)
        speed["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)

    metrics = {
        "weights": str(args.weights),
        "model": ckpt["model_name"],
        "epoch": ckpt["epoch"],
        "eval_min_size": ckpt["min_size"][-1],
        "score_thresh": model.roi_heads.score_thresh,
        "split_json": str(args.json),
        "n_images": len(dataset),
        "n_detections": len(dets),
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "coco": stats,
        "pr_iou50": pr,
        "speed": speed,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "per_class_ap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["category", "n_gt", "AP50_95", "AP50"])
        w.writeheader()
        w.writerows({**r, "AP50_95": f"{r['AP50_95']:.4f}", "AP50": f"{r['AP50']:.4f}"} for r in rows)

    b = pr["best_f1"] or {}
    print(f"\nmAP@.5:.95 {stats['mAP50_95']:.4f} | mAP@.5 {stats['mAP50']:.4f} | "
          f"P {b.get('precision', 0):.4f} R {b.get('recall', 0):.4f} (conf {b.get('conf', 0):.3f}, best F1)")
    if speed:
        print(f"speed: {speed['ms_per_img_mean']:.1f} ms/img ({speed['fps']:.1f} FPS) on {speed['device']}")
    print("\nper-class AP@.5:.95 / AP@.5:")
    for r in rows:
        print(f"  {r['category']:<24} n={r['n_gt']:<5} {r['AP50_95']:.3f} / {r['AP50']:.3f}")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()