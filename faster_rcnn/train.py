"""Fine-tune a COCO-pretrained torchvision Faster R-CNN on TrashCan (COCO-format json).

Example (Windows, RTX 5070):
    python train.py --train-json D:/debris/data/instances_train.json --train-images D:/debris/data/train ^
                    --val-json D:/debris/data/instances_val.json --val-images D:/debris/data/val ^
                    --out D:/debris/runs/frcnn_r50v2_800
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from common import (MODELS, CocoDetection, build_model, collate_fn, get_transforms, load_categories,
                    pick_device, save_checkpoint, set_seed, take_subset)
from metrics import autocast, coco_evaluate, load_coco, plot_training_curves, run_inference

LOSS_KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-json", required=True)
    p.add_argument("--train-images", required=True)
    p.add_argument("--val-json", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--out", default="runs/frcnn", help="run folder (keep it outside OneDrive)")
    p.add_argument("--model", default="resnet50_fpn_v2", choices=list(MODELS))
    p.add_argument("--pretrained", default="coco", choices=["coco", "imagenet", "none"])
    p.add_argument("--trainable-layers", type=int, default=3, help="backbone stages to fine-tune (0 = frozen)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=None, help="default: 0.02 * batch_size / 16")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--min-size", type=int, nargs="+", default=[800],
                   help="shorter side after resize; several values = multi-scale training, last one used for eval")
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--aug", default="basic", choices=["none", "basic", "strong"],
                   help="basic = horizontal flip, strong = flip + photometric distortion")
    p.add_argument("--mosaic", type=float, default=0.0,
                   help="probability of a YOLO-style 4-frame mosaic per training image, 0 = off")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--subset", type=int, default=0, help="use only the first N train/val images (quick test)")
    p.add_argument("--resume", default=None, help="path to last.pt to continue training")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=50)
    return p.parse_args()


def lr_lambda(warmup, total, min_factor=0.01):
    """Linear warmup, then cosine decay to min_factor * lr (per iteration)."""
    def f(it):
        if it < warmup:
            return 0.001 + (1 - 0.001) * it / warmup
        progress = (it - warmup) / max(1, total - warmup)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return f


def make_loader(dataset, batch_size, shuffle, workers, device):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0)


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = 0.02 * args.batch_size / 16
    set_seed(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    categories = load_categories(args.train_json)
    val_ids = [c["id"] for c in load_categories(args.val_json)]
    if val_ids != [c["id"] for c in categories]:
        raise SystemExit("train and val json have different categories - use the same TrashCan version")
    label_to_cat = {i + 1: c["id"] for i, c in enumerate(categories)}

    train_set = CocoDetection(args.train_json, args.train_images, categories, get_transforms(True, args.aug),
                              mosaic=args.mosaic)
    val_set = CocoDetection(args.val_json, args.val_images, categories, get_transforms(False))
    coco_val = load_coco(args.val_json)
    if args.subset:
        take_subset(train_set, None, args.subset)
        take_subset(val_set, coco_val, args.subset)
    train_loader = make_loader(train_set, args.batch_size, True, args.workers, device)
    val_loader = make_loader(val_set, args.batch_size, False, args.workers, device)

    model = build_model(args.model, len(categories), args.pretrained, args.trainable_layers,
                        args.min_size, args.max_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    total_iters = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(args.warmup_iters, total_iters))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_map = 1, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_map = ckpt["epoch"] + 1, ckpt["best_map"]
        print(f"resumed from {args.resume} at epoch {ckpt['epoch']}")
    else:
        (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in params)
    print(f"device={device} amp={use_amp} model={args.model} pretrained={args.pretrained} "
          f"classes={len(categories)} params={n_total / 1e6:.1f}M (trainable {n_train / 1e6:.1f}M)")
    print(f"train images={len(train_set)} val images={len(val_set)} iters/epoch={len(train_loader)} lr={args.lr}")

    results_path = out / "results.csv"
    fields = ["epoch", "lr", "loss", *LOSS_KEYS, "mAP50_95", "mAP50", "mAP75", "mAP_small", "mAP_medium",
              "mAP_large", "AR100", "epoch_min", "val_min"]
    if not results_path.exists():
        with open(results_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()
    iter_log = open(out / "iter_log.csv", "a", newline="")
    iter_writer = csv.writer(iter_log)
    if iter_log.tell() == 0:
        iter_writer.writerow(["epoch", "iter", "global_iter", "lr", "loss", *LOSS_KEYS])

    train_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums = {k: 0.0 for k in ["loss", *LOSS_KEYS]}
        for it, (imgs, targets) in enumerate(train_loader, 1):
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{"boxes": t["boxes"].as_subclass(torch.Tensor).to(device),
                        "labels": t["labels"].to(device)} for t in targets]
            with autocast(device, use_amp):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise SystemExit(f"loss is {loss_value} at epoch {epoch} iter {it}: "
                                 f"{ {k: v.item() for k, v in loss_dict.items()} } - try a lower --lr")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            parts = {k: loss_dict[k].item() for k in LOSS_KEYS}
            sums["loss"] += loss_value
            for k in LOSS_KEYS:
                sums[k] += parts[k]
            if it % args.print_every == 0 or it == len(train_loader):
                lr = optimizer.param_groups[0]["lr"]
                g_it = (epoch - 1) * len(train_loader) + it
                iter_writer.writerow([epoch, it, g_it, f"{lr:.6g}", f"{loss_value:.4f}",
                                      *(f"{parts[k]:.4f}" for k in LOSS_KEYS)])
                iter_log.flush()
                eta = (time.time() - t0) / it * (len(train_loader) - it)
                mem = f" mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G" if device.type == "cuda" else ""
                print(f"epoch {epoch}/{args.epochs} iter {it}/{len(train_loader)} lr={lr:.5f} "
                      f"loss={sums['loss'] / it:.4f} cls={sums['loss_classifier'] / it:.3f} "
                      f"box={sums['loss_box_reg'] / it:.3f} obj={sums['loss_objectness'] / it:.3f} "
                      f"rpn={sums['loss_rpn_box_reg'] / it:.3f} eta={eta / 60:.1f}min{mem}", flush=True)
        epoch_min = (time.time() - t0) / 60

        row = {"epoch": epoch, "lr": f"{optimizer.param_groups[0]['lr']:.6g}", "epoch_min": f"{epoch_min:.2f}"}
        row.update({k: f"{v / len(train_loader):.4f}" for k, v in sums.items()})
        stats = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            t1 = time.time()
            # FP32 like evaluate.py: under autocast the box decoder rounds coordinates to float16.
            dets = run_inference(model, val_loader, device, label_to_cat)
            stats, _ = coco_evaluate(coco_val, dets)
            row.update({k: f"{stats[k]:.4f}" for k in fields if k in stats})
            row["val_min"] = f"{(time.time() - t1) / 60:.2f}"
            print(f"epoch {epoch} val mAP@.5:.95={stats['mAP50_95']:.4f} mAP@.5={stats['mAP50']:.4f} "
                  f"({len(dets)} detections, {epoch_min:.1f} min train)", flush=True)
        with open(results_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        if stats is not None and stats["mAP50_95"] > best_map:
            best_map = stats["mAP50_95"]
            save_checkpoint(out / "best.pt", model, args, categories, epoch, best_map)
            print(f"  new best -> {out / 'best.pt'}")
        save_checkpoint(out / "last.pt", model, args, categories, epoch, best_map, optimizer, scheduler, scaler)
        plot_training_curves(out)

    iter_log.close()
    summary = {"best_val_mAP50_95": best_map, "epochs": args.epochs, "train_hours": (time.time() - train_start) / 3600,
               "params_M": n_total / 1e6, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done. best val mAP@.5:.95 = {best_map:.4f}")
    print(f"next: python evaluate.py --weights {out / 'best.pt'} --json <test json> --images <test images> "
          f"--out {out / 'eval_test'}")


if __name__ == "__main__":
    main()