"""Draw Faster R-CNN predictions on images (for the report / slides).

With --json the ground truth is drawn next to the prediction (GT | prediction),
which makes missed and false detections easy to see.

    python predict.py --weights runs/frcnn/best.pt --images D:/debris/data/val --json D:/debris/data/instances_val.json --out vis
"""

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor

from common import load_model_from_checkpoint, pick_device

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True, help="image folder")
    p.add_argument("--json", default=None, help="optional GT json: pick images from it and draw GT alongside")
    p.add_argument("--files", nargs="*", default=None, help="specific file names inside --images")
    p.add_argument("--out", default="vis")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--num", type=int, default=16, help="random images to draw when --files is not given")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def get_font(size=12):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def color(i):
    rng = random.Random(i * 7919)
    return tuple(rng.randint(60, 255) for _ in range(3))


def draw_boxes(img, boxes, names, cat_ids, scores=None, font=None):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for k, (box, cid) in enumerate(zip(boxes, cat_ids)):
        c = color(cid)
        d.rectangle(box, outline=c, width=2)
        text = names[cid] if scores is None else f"{names[cid]} {scores[k]:.2f}"
        x, y = box[0], max(0, box[1] - 14)
        tb = d.textbbox((x, y), text, font=font)
        d.rectangle(tb, fill=c)
        d.text((x, y), text, fill=(0, 0, 0), font=font)
    return img


def titled(img, title, font):
    canvas = Image.new("RGB", (img.width, img.height + 18), (255, 255, 255))
    canvas.paste(img, (0, 18))
    ImageDraw.Draw(canvas).text((4, 2), title, fill=(0, 0, 0), font=font)
    return canvas


def main():
    args = parse_args()
    device = pick_device(args.device)
    model, categories, _ = load_model_from_checkpoint(args.weights, device)
    names = {c["id"]: c["name"] for c in categories}
    label_to_cat = {i + 1: c["id"] for i, c in enumerate(categories)}
    font = get_font()
    image_dir, out = Path(args.images), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt = {}
    if args.json:
        with open(args.json, encoding="utf-8") as f:
            data = json.load(f)
        by_id = {im["id"]: im["file_name"] for im in data["images"]}
        for name in by_id.values():
            gt[name] = []
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            gt[by_id[a["image_id"]]].append(([x, y, x + w, y + h], a["category_id"]))
        pool = sorted(gt)
    else:
        pool = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    files = args.files or random.Random(args.seed).sample(pool, min(args.num, len(pool)))

    panels = []
    for name in files:
        img = Image.open(image_dir / name).convert("RGB")
        x = pil_to_tensor(img).float().div(255).to(device)
        with torch.inference_mode():
            o = model([x])[0]
        keep = o["scores"] >= args.conf
        boxes = o["boxes"][keep].cpu().tolist()
        cids = [label_to_cat[l] for l in o["labels"][keep].cpu().tolist()]
        pred = draw_boxes(img, boxes, names, cids, o["scores"][keep].cpu().tolist(), font)
        pred = titled(pred, f"Faster R-CNN (conf>={args.conf}): {len(boxes)} boxes", font)
        if args.json:
            g = gt.get(name, [])
            gimg = titled(draw_boxes(img, [b for b, _ in g], names, [c for _, c in g], font=font),
                          f"Ground truth: {len(g)} boxes", font)
            panel = Image.new("RGB", (gimg.width + pred.width + 6, gimg.height), (255, 255, 255))
            panel.paste(gimg, (0, 0))
            panel.paste(pred, (gimg.width + 6, 0))
        else:
            panel = pred
        panel.save(out / f"{Path(name).stem}_pred.jpg", quality=92)
        panels.append(panel)

    # One montage of the first 8 panels, handy as a report figure.
    grid = panels[:8]
    if grid:
        w, h = grid[0].size
        cols = 1 if args.json else 2
        rows = (len(grid) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
        for i, p in enumerate(grid):
            sheet.paste(p.resize((w, h)), ((i % cols) * w, (i // cols) * h))
        sheet.save(out / "montage.jpg", quality=92)
    print(f"saved {len(panels)} images + montage.jpg to {out}")


if __name__ == "__main__":
    main()