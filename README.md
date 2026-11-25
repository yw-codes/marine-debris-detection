# Underwater Marine Debris Detection: YOLO11n vs Faster R-CNN

Object detection of marine debris, marine life and ROV parts in underwater footage, using the 22-class
[TrashCan 1.0 (instance)](https://conservancy.umn.edu/handle/11299/214865) dataset.
We compare a compact one-stage detector (**YOLO11n**) with a two-stage detector (**Faster R-CNN**)
on a **video-disjoint** split, so that no test frame comes from a video seen during training.

Group project for UNSW COMP9444 *Neural Networks and Deep Learning* (2026 T3), team **Deep Thinkers**.

## My contribution

I built and ran the **Faster R-CNN** pipeline (the code in [`faster_rcnn/`](faster_rcnn)):

- Fine-tuned torchvision's COCO-pretrained Faster R-CNN ResNet-50-FPN v2 with a new 22-class head,
  fine-tuning the last three backbone stages and freezing BatchNorm statistics (batch size 4 on an 8 GB RTX 5070 laptop GPU).
- Ran the ablations on pretraining (COCO vs ImageNet), input size (480 vs 800 px) and backbone (ResNet-50 vs MobileNetV3).
- Wrote the evaluation tooling (COCO mAP via pycocotools, precision/recall, inference speed, training curves)
  and a sanity check that scores the ground truth against itself.

## Results (held-out test set: 47 unseen videos, 993 frames)

Both models are scored with the same pycocotools protocol.

| Metric | YOLO11n | Faster R-CNN |
|---|---:|---:|
| mAP@[.5:.95] | 0.073 | **0.150** |
| mAP@.5 | 0.114 | **0.233** |
| Recall | 0.186 | **0.327** |
| AR@100 | 0.309 | **0.510** |
| Parameters | **2.6 M** | 43.4 M |

Faster R-CNN ablations (test mAP@[.5:.95]):

| Run | Change | mAP | Latency |
|---|---|---:|---:|
| `r50v2_coco_800` | reference | 0.110 | 65.4 ms/img |
| `r50v2_imagenet_800` | ImageNet-only backbone | 0.060 | 65.0 ms/img |
| **`r50v2_coco_480`** | native 480 px input (final model) | **0.150** | 39.5 ms/img |
| `mnv3_coco_800` | MobileNetV3 backbone | 0.111 | 12.8 ms/img |

**Why the scores are lower than the published TrashCan baseline:** the original split shares videos between
train and test. Retraining the same configuration on the original split reaches **0.476** validation mAP
(published X-101 baseline: 0.345), so the gap comes from the stricter video-disjoint evaluation, not from under-training.

The full analysis (per-class and per-size AP, error breakdown, colour-cast robustness, qualitative examples)
is in the [notebook](notebook/marine_debris_detection.ipynb) and the [report](report/project_report.pdf).

## Repository layout

```
faster_rcnn/
  common.py          dataset (COCO json), transforms, model builder, checkpoints
  train.py           fine-tuning script (SGD, warm-up, AMP, best-checkpoint selection)
  evaluate.py        test-set evaluation -> metrics.json, per-class AP, plots
  metrics.py         inference to COCO json, COCO mAP, P/R, speed, curves
  predict.py         draws predictions (optionally next to ground truth)
  verify_metrics.py  scores ground truth against itself (mAP, P, R should be ~1.0)
notebook/            full team notebook with saved outputs (data split, YOLO, Faster R-CNN, comparison)
report/              final project report
```

## Running the Faster R-CNN pipeline

The dataset and model weights are not included. Download TrashCan 1.0 and create the
video-disjoint split as described in Section 3 of the notebook, then:

```bash
pip install -r requirements.txt
cd faster_rcnn

python train.py --train-json data/instances_train.json --train-images data/train \
                --val-json data/instances_val.json --val-images data/val \
                --min-size 480 --max-size 854 --out runs/frcnn_r50v2_480

python evaluate.py --help     # evaluate a checkpoint on the test split
```

`python train.py --help` lists every option (backbone, pretraining, trainable stages, augmentation, AMP, resume).

## Team

Junfeng He, Yu Lok Fu, Fei Fang and Yu Wang.
