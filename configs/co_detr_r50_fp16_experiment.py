"""
co_detr_r50_fp16_experiment.py — Konfigurasi mmdetection untuk Kondisi A (FP16)

Config ini mewarisi dari config Co-DETR official (co_dino_5scale_r50_1x_coco.py
atau yang paling ringan tersedia), lalu override untuk:
  - FP16 training dengan dynamic loss scaling
  - Dataset: COCO subset 500 gambar
  - Max iter: 300 (cukup untuk membuktikan overflow)
  - Batch size: 2
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────
# Base config dari repo Co-DETR
# Sesuaikan path ini dengan lokasi repo yang sudah di-clone
# ─────────────────────────────────────────────
_base_ = [
    # Coba urutan ini secara berurutan — gunakan yang pertama tersedia
    # Uncomment salah satu:
    "../Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py",
    # "../Co-DETR/configs/co_detr/co_detr_r50_fpn_1x_coco.py",
]

# ─────────────────────────────────────────────
# Dataset: COCO subset
# ─────────────────────────────────────────────
data_root = os.environ.get("CODETR_DATA_ROOT", "data/coco_subset")

dataset_type = "CocoDataset"

# Override train dataset dengan subset
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
# Training schedule: sangat pendek (bukti overflow, bukan akurasi)
# ─────────────────────────────────────────────
runner = dict(
    type="IterBasedRunner",
    max_iters=300,  # cukup untuk memicu overflow pertama
)

checkpoint_config = dict(interval=300)  # simpan checkpoint hanya di akhir

log_config = dict(
    interval=5,  # log setiap 5 iter untuk tracking yang baik
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

# ─────────────────────────────────────────────
# ⚡ FP16: KUNCI KONDISI A ⚡
# Dynamic loss scaling aktif — GradScaler akan log setiap overflow
# ─────────────────────────────────────────────
fp16 = dict(
    loss_scale=dict(
        init_scale=65536,      # initial scale = 2^16
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,  # naikkan scale setiap 2000 iter jika tidak ada overflow
    )
)

# ─────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=1e-4,
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# ─────────────────────────────────────────────
# TIDAK ada modifikasi arsitektur (untuk membuktikan overflow)
# JANGAN tambahkan:
#   - QK-normalization
#   - logit clamping
#   - gradient clipping yang terlalu agresif
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Catatan untuk eksperimen
# ─────────────────────────────────────────────
# Kondisi A sengaja rentan terhadap overflow:
#   1. FP16 dynamic loss scaling aktif
#   2. Tidak ada modifikasi arsitektur pelindung
#   3. Auxiliary heads (ATSS/FCOS) dengan skala loss berbeda
#   4. Dataset dengan gambar banyak objek (memicu logit besar)
