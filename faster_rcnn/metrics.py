"""Evaluation helpers: inference to COCO json, COCO mAP, YOLO-style P/R, speed, curves."""

import contextlib
import csv
import io
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_NAMES = ["mAP50_95", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def load_coco(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def autocast(device, enabled):
    return torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.float16,
                          enabled=enabled and device.type == "cuda")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def run_inference(model, loader, device, label_to_cat, amp=False):
    """Detections for every image in `loader`, as a COCO results list (xywh boxes, original ids)."""
    model.eval()
    dets = []
    for imgs, targets in loader:
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        with autocast(device, amp):
            outs = model(imgs)
        for out, t in zip(outs, targets):
            boxes = out["boxes"].float().cpu()
            boxes[:, 2:] -= boxes[:, :2]
            for b, s, l in zip(boxes.tolist(), out["scores"].tolist(), out["labels"].tolist()):
                dets.append({"image_id": t["image_id"], "category_id": label_to_cat[l],
                             "bbox": [round(v, 2) for v in b], "score": round(s, 5)})
    return dets


def coco_evaluate(coco_gt, dets, verbose=False):
    """COCO bbox metrics. Returns (stats dict, COCOeval or None when there are no detections)."""
    if not dets:
        return {k: 0.0 for k in STAT_NAMES}, None
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        # loadRes adds fields to each dict, so give it copies.
        coco_dt = coco_gt.loadRes([dict(d) for d in dets])
        E = COCOeval(coco_gt, coco_dt, "bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
    if verbose:
        print(out.getvalue().split("DONE")[-1].strip())
    return dict(zip(STAT_NAMES, (float(s) for s in E.stats))), E


def per_class_ap(E, coco_gt, categories):
    """AP@[.5:.95] and AP@.5 per category from an accumulated COCOeval."""
    n_gt = defaultdict(int)
    for a in coco_gt.dataset["annotations"]:
        if not a.get("iscrowd", 0):
            n_gt[a["category_id"]] += 1
    rows = []
    precision = E.eval["precision"] if E is not None else None  # [IoU, recall, class, area, maxDets]
    for c in categories:
        ap = ap50 = float("nan")
        if precision is not None and c["id"] in E.params.catIds:
            k = E.params.catIds.index(c["id"])
            p = precision[:, :, k, 0, -1]
            p50 = precision[0, :, k, 0, -1]
            ap = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
            ap50 = float(p50[p50 > -1].mean()) if (p50 > -1).any() else float("nan")
        rows.append({"category": c["name"], "n_gt": n_gt[c["id"]], "AP50_95": ap, "AP50": ap50})
    return rows


def _iou(a, b):
    """IoU between xyxy arrays a [N,4] and b [M,4]."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def precision_recall(coco_gt, dets, categories, iou_thr=0.5, fixed_conf=0.5):
    """Precision / recall at IoU 0.5, computed the way Ultralytics reports them for YOLO.

    Detections are matched greedily per class by descending score. P and R are
    averaged over the classes present in the ground truth, at the confidence
    threshold that maximises the mean F1 (and also at `fixed_conf`).
    """
    gts, dts = defaultdict(list), defaultdict(list)
    for a in coco_gt.dataset["annotations"]:
        if not a.get("iscrowd", 0):
            gts[(a["image_id"], a["category_id"])].append(_xyxy(a["bbox"]))
    for d in dets:
        dts[(d["image_id"], d["category_id"])].append((d["score"], _xyxy(d["bbox"])))

    scores, tps, n_gt = defaultdict(list), defaultdict(list), defaultdict(int)
    for key in set(gts) | set(dts):
        cat = key[1]
        g = np.array(gts.get(key, []), dtype=np.float64).reshape(-1, 4)
        d = sorted(dts.get(key, []), key=lambda x: -x[0])
        n_gt[cat] += len(g)
        if not d:
            continue
        db = np.array([b for _, b in d], dtype=np.float64)
        iou = _iou(db, g) if len(g) else np.zeros((len(d), 0))
        matched = np.zeros(len(g), dtype=bool)
        for i, (s, _) in enumerate(d):
            tp = False
            if len(g):
                cand = np.where(matched, -1.0, iou[i])
                j = int(cand.argmax())
                if cand[j] >= iou_thr:
                    matched[j] = tp = True
            scores[cat].append(s)
            tps[cat].append(tp)

    conf = np.linspace(0, 1, 1001)
    cats = [c["id"] for c in categories if n_gt[c["id"]] > 0]
    P = np.ones((len(cats), len(conf)))
    R = np.zeros((len(cats), len(conf)))
    for k, cat in enumerate(cats):
        s = np.array(scores[cat])
        if len(s) == 0:
            continue
        order = np.argsort(-s)
        s = s[order]
        cum_tp = np.concatenate([[0], np.cumsum(np.array(tps[cat])[order])])
        n = np.searchsorted(-s, -conf, side="right")  # detections with score >= conf
        P[k] = np.where(n > 0, cum_tp[n] / np.maximum(n, 1), 1.0)
        R[k] = cum_tp[n] / n_gt[cat]
    F1 = 2 * P * R / (P + R + 1e-16)

    def at(i):
        return {"precision": float(P[:, i].mean()), "recall": float(R[:, i].mean()),
                "f1": float(F1[:, i].mean()), "conf": float(conf[i])}

    if not cats:
        return {"best_f1": None, "fixed_conf": None}
    best = int(F1.mean(0).argmax())
    fixed = int(np.abs(conf - fixed_conf).argmin())
    return {"best_f1": at(best), "fixed_conf": at(fixed)}


@torch.inference_mode()
def benchmark(model, dataset, device, n=200, warmup=20, amp=False):
    """Model forward time per image at batch size 1 (resize + network + NMS; JPEG decode excluded)."""
    model.eval()
    times = []
    for k in range(warmup + n):
        img = dataset[k % len(dataset)][0].to(device)
        synchronize(device)
        t0 = time.perf_counter()
        with autocast(device, amp):
            model([img])
        synchronize(device)
        if k >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"ms_per_img_mean": float(t.mean()), "ms_per_img_median": float(np.median(t)),
            "fps": float(1000 / t.mean()), "n_images": n, "batch_size": 1, "amp": bool(amp)}


def plot_training_curves(run_dir):
    """Loss components and val mAP per epoch from results.csv -> curves.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    with open(run_dir / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    col = lambda k: [float(r[k]) if r[k] != "" else np.nan for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k in ["loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        axes[0].plot(ep, col(k), label=k, lw=2 if k == "loss" else 1.2)
    axes[0].set(title="Training loss", xlabel="epoch")
    axes[0].legend(fontsize=8)
    for k, lab in [("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95")]:
        axes[1].plot(ep, col(k), marker="o", ms=3, label=lab)
    axes[1].set(title="Validation mAP", xlabel="epoch")
    axes[1].legend(fontsize=8)
    axes[2].plot(ep, col("lr"))
    axes[2].set(title="Learning rate (end of epoch)", xlabel="epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=150)
    plt.close(fig)


def plot_pr_curves(E, categories, out_path):
    """Per-class precision-recall curves at IoU 0.5 (thin) and their mean (bold)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if E is None:
        return
    rec = E.params.recThrs
    prec = E.eval["precision"][0, :, :, 0, -1]  # [recall, class]
    fig, ax = plt.subplots(figsize=(7, 6))
    valid = []
    for c in categories:
        if c["id"] not in E.params.catIds:
            continue
        p = prec[:, E.params.catIds.index(c["id"])]
        if (p > -1).any():
            ax.plot(rec, p, lw=0.8, alpha=0.6, label=f"{c['name']} {p.mean():.2f}")
            valid.append(p)
    if valid:
        mean = np.mean(valid, 0)
        ax.plot(rec, mean, lw=3, color="black", label=f"all classes {mean.mean():.3f} mAP@0.5")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Precision-Recall (IoU=0.5)")
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)