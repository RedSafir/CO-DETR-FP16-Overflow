"""
co_detr_r50_fp32_baseline.py — Konfigurasi mmdetection untuk Kondisi B (FP32 Baseline)

Identik dengan Kondisi A, tapi TANPA fp16 setting.
Digunakan sebagai pembanding "tidak ada overflow".
"""
import os

# ─────────────────────────────────────────────
# Base config identik dengan Kondisi A
# ─────────────────────────────────────────────
_base_ = [
    "../Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py",
]

# ─────────────────────────────────────────────
# Dataset: identik dengan Kondisi A
# ─────────────────────────────────────────────
data_root = os.environ.get("CODETR_DATA_ROOT", "data/coco_subset")

dataset_type = "CocoDataset"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_val2017_subset.json",
        img_prefix="images/val2017/",
        filter_empty_gt=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_val2017_subset.json",
        img_prefix="images/val2017/",
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_val2017_subset.json",
        img_prefix="images/val2017/",
    ),
)

# ─────────────────────────────────────────────
# Training schedule: identik dengan Kondisi A
# ─────────────────────────────────────────────
runner = dict(
    type="IterBasedRunner",
    max_iters=300,
)

checkpoint_config = dict(interval=300)

log_config = dict(
    interval=5,
    hooks=[dict(type="TextLoggerHook")],
)

# ─────────────────────────────────────────────
# ✅ FP32: TIDAK ADA fp16 setting
# Ini adalah pembanding — tanpa dynamic loss scaling
# ─────────────────────────────────────────────
# fp16 = ...  ← TIDAK ADA di Kondisi B

# ─────────────────────────────────────────────
# Optimizer identik
# ─────────────────────────────────────────────
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=1e-4,
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# ─────────────────────────────────────────────
# Catatan:
# Kondisi B sebagai "ground truth" — nilai max_abs_value seharusnya
# tetap di bawah 65504 karena FP32 punya range jauh lebih luas (~3.4e38)
# ─────────────────────────────────────────────
