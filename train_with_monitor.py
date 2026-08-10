"""
train_with_monitor.py — Training loop Co-DETR dengan instrumentasi overflow
Versi yang kompatibel dengan mmdetection v2.x dan v3.x (OpenMMLab 2.0)

Penggunaan:
  # Kondisi A — FP16
  python train_with_monitor.py \
    --condition fp16 \
    --max-iters 300 \
    --log-dir results/conditionA_fp16 \
    --data-root data/coco_subset

  # Kondisi B — FP32
  python train_with_monitor.py \
    --condition fp32 \
    --max-iters 300 \
    --log-dir results/conditionB_fp32 \
    --data-root data/coco_subset
"""
import argparse
import csv
import math
import os
import sys
import time
import traceback
from pathlib import Path

# ─────────────────────────────────────────────
# Setup path
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent
CODETR_DIR = ROOT / "Co-DETR"

if CODETR_DIR.exists():
    sys.path.insert(0, str(CODETR_DIR))
    # Tambahkan projects/ ke path untuk custom modules
    projects_dir = CODETR_DIR / "projects"
    if projects_dir.exists():
        sys.path.insert(0, str(projects_dir))

sys.path.insert(0, str(ROOT / "instrumentation"))

import torch
import torch.nn as nn
from overflow_monitor import OverflowMonitor, LoggingGradScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default=None, help="Path ke config mmdetection")
    p.add_argument("--condition",   required=True, choices=["fp16", "fp32"])
    p.add_argument("--max-iters",   type=int,   default=300)
    p.add_argument("--log-dir",     required=True)
    p.add_argument("--data-root",   default="data/coco_subset")
    p.add_argument("--batch-size",  type=int,   default=2)
    p.add_argument("--strict-fp16", action="store_true",
                   help="Patch attention untuk FP16 murni (tanpa autocast protection)")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def find_codetr_config(given_config: str = None) -> str:
    """Cari config Co-DETR yang tersedia di repo."""
    if given_config and Path(given_config).exists():
        print(f"  [OK] Using specified config: {given_config}")
        return given_config

    candidates = [
        "configs/co_detr_r50_fp16_experiment.py",
        "Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py",
        "Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_lsj_1x_coco.py",
        "Co-DETR/configs/co_detr/co_detr_r50_fpn_1x_coco.py",
        "Co-DETR/projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py",
    ]
    for cfg in candidates:
        if Path(cfg).exists():
            print(f"  [OK] Config ditemukan: {cfg}")
            return cfg
    raise FileNotFoundError(
        "Tidak ada config Co-DETR yang ditemukan!\n"
        f"Dicari di: {candidates}"
    )


def build_mmdet_components(args, config_path: str):
    """
    Build model, dataset, dataloader menggunakan mmdetection API.
    Kompatibel dengan mmdet v2.x dan v3.x.
    """
    # ── Deteksi versi mmdet ──────────────────────────────────────────────
    import mmdet
    mmdet_major = int(mmdet.__version__.split(".")[0])
    print(f"  mmdetection version: {mmdet.__version__} (v{mmdet_major}.x)")

    if mmdet_major >= 3:
        return _build_mmdet_v3(args, config_path)
    else:
        return _build_mmdet_v2(args, config_path)


def _build_mmdet_v2(args, config_path: str):
    """Build dengan mmdetection v2.x API."""
    try:
        from mmcv import Config
    except ImportError:
        try:
            from mmengine.config import Config
        except ImportError:
            from mmengine import Config

    from mmdet.models import build_detector
    from mmdet.datasets import build_dataset, build_dataloader

    cfg = Config.fromfile(config_path)

    # Override data
    data_root = Path(args.data_root).resolve()
    cfg.data.train.data_root   = str(data_root)
    cfg.data.train.ann_file    = "annotations/instances_val2017_subset.json"
    cfg.data.train.img_prefix  = "images/val2017/"
    cfg.data.samples_per_gpu   = args.batch_size
    cfg.data.workers_per_gpu   = 2

    # Override training schedule
    cfg.runner = dict(type="IterBasedRunner", max_iters=args.max_iters)
    cfg.log_config = dict(interval=10, hooks=[dict(type="TextLoggerHook")])
    cfg.checkpoint_config = dict(interval=9999)  # jangan simpan checkpoint

    # FP16 config
    if args.condition == "fp16":
        cfg.fp16 = dict(loss_scale=dict(
            init_scale=65536, growth_factor=2.0,
            backoff_factor=0.5, growth_interval=2000,
        ))
    elif hasattr(cfg, "fp16"):
        del cfg.fp16

    # Build model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    model.cuda()
    model.train()

    # Build dataset
    dataset = build_dataset(cfg.data.train)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=2,
        num_gpus=1,
        dist=False,
        shuffle=True,
        seed=args.seed,
    )

    return model, dataloader, cfg


def _build_mmdet_v3(args, config_path: str):
    """Build dengan mmdetection v3.x (OpenMMLab 2.0) API."""
    from mmengine.config import Config
    from mmdet.registry import MODELS, DATASETS
    from mmengine.registry import DATASETS as ENGINE_DATASETS
    from torch.utils.data import DataLoader

    cfg = Config.fromfile(config_path)

    data_root = str(Path(args.data_root).resolve())

    # Override dataset config (v3.x style)
    if hasattr(cfg, "train_dataloader"):
        cfg.train_dataloader.batch_size = args.batch_size
        cfg.train_dataloader.num_workers = 2
        if hasattr(cfg.train_dataloader, "dataset"):
            cfg.train_dataloader.dataset.data_root = data_root
            cfg.train_dataloader.dataset.ann_file = \
                "annotations/instances_val2017_subset.json"
            cfg.train_dataloader.dataset.data_prefix = \
                dict(img="images/val2017/")

    # Build model (v3.x)
    model = MODELS.build(cfg.model)
    model.cuda()
    model.train()

    # Build dataset & loader (v3.x)
    dataset_cfg = cfg.train_dataloader.dataset if hasattr(cfg, "train_dataloader") else cfg.data.train
    try:
        dataset = DATASETS.build(dataset_cfg)
    except Exception:
        from mmdet.datasets import build_dataset
        dataset = build_dataset(dataset_cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    return model, dataloader, cfg


def _collate_fn(batch):
    """Custom collate untuk mmdetection DataInfo format."""
    # Coba pakai mmdet collate
    try:
        from mmdet.utils import collate
        return collate(batch, samples_per_gpu=len(batch))
    except Exception:
        pass
    try:
        from mmcv.parallel import collate
        return collate(batch, samples_per_gpu=len(batch))
    except Exception:
        pass
    # Fallback: PyTorch default
    import torch
    return torch.utils.data.dataloader.default_collate(batch)


def build_optimizer(model, cfg):
    """Build optimizer dari config."""
    # Coba berbagai API
    try:
        from mmdet.core import build_optimizer
        return build_optimizer(model, cfg.optimizer)
    except (ImportError, AttributeError):
        pass
    try:
        from mmengine.optim import build_optim_wrapper
        return build_optim_wrapper(model, cfg.optim_wrapper)
    except (ImportError, AttributeError):
        pass
    # Fallback: AdamW default
    print("  [WARN] Gagal build optimizer dari config — pakai AdamW default.")
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )


def forward_model(model, batch, condition):
    """
    Forward pass yang kompatibel dengan mmdet v2 dan v3.
    """
    # mmdet v2: model(**batch) → dict of losses
    # mmdet v3: model.loss(batch) atau model(**batch)
    if isinstance(batch, dict):
        if condition == "fp16":
            with torch.cuda.amp.autocast():
                output = model(**batch)
        else:
            output = model(**batch)
    elif isinstance(batch, (list, tuple)):
        # mmdet v2 format: list of data_samples
        imgs = batch[0].cuda() if hasattr(batch[0], "cuda") else batch[0]
        img_metas = batch[1] if len(batch) > 1 else []
        if condition == "fp16":
            with torch.cuda.amp.autocast():
                output = model(img=imgs, img_metas=img_metas, return_loss=True)
        else:
            output = model(img=imgs, img_metas=img_metas, return_loss=True)
    else:
        raise ValueError(f"Tipe batch tidak dikenali: {type(batch)}")

    return output


def aggregate_loss(output):
    """Jumlahkan semua komponen loss dari output model."""
    if isinstance(output, dict):
        loss_total = sum(
            v for k, v in output.items()
            if isinstance(v, torch.Tensor) and v.requires_grad
        )
        loss_dict = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in output.items()
        }
        return loss_total, loss_dict
    elif isinstance(output, torch.Tensor):
        return output, {"loss": output.item()}
    else:
        raise ValueError(f"Output model tidak dikenali: {type(output)}")


def run_training(args):
    """Main training loop."""
    print("\n" + "="*60)
    print(f"  Co-DETR FP16 OVERFLOW EXPERIMENT")
    print(f"  Kondisi : {args.condition.upper()}")
    print(f"  Max iter: {args.max_iters}")
    print(f"  Log dir : {args.log_dir}")
    print("="*60)

    torch.manual_seed(args.seed)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Find config ────────────────────────────────────────────────────
    print("\n[1/5] Mencari config Co-DETR...")
    config_path = find_codetr_config(args.config)

    # ── 2. Build model & dataloader ───────────────────────────────────────
    print("\n[2/5] Build model dan dataloader...")
    try:
        model, dataloader, cfg = build_mmdet_components(args, config_path)
    except Exception as e:
        print(f"  [ERROR] Gagal build model: {e}")
        traceback.print_exc()
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {type(model).__name__} ({n_params:.1f}M params)")

    # ── 3. Setup monitoring ────────────────────────────────────────────────
    print("\n[3/5] Setup overflow monitoring hooks...")
    monitor = OverflowMonitor(
        log_dir=str(log_dir),
        condition=args.condition,
        log_every=1,
        verbose=True,
    )
    monitor.register_all_hooks(model)

    if args.strict_fp16 and args.condition == "fp16":
        from overflow_monitor import patch_mha_for_strict_fp16
        patched = patch_mha_for_strict_fp16(model, monitor)
        print(f"  Strict FP16 patch: {patched} attention modules patched")

    # ── 4. Optimizer & Scaler ─────────────────────────────────────────────
    print("\n[4/5] Setup optimizer...")
    optimizer = build_optimizer(model, cfg)

    scaler = None
    if args.condition == "fp16":
        scaler = LoggingGradScaler(
            log_path=str(log_dir / "gradscaler_log.csv"),
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )
        print(f"  GradScaler: scale={scaler.get_scale():.0f}")

    # ── 5. Training loop ──────────────────────────────────────────────────
    print("\n[5/5] Training loop dimulai...\n")
    training_log = []
    step = 0
    epoch = 0
    t_start = time.time()

    try:
        while step < args.max_iters:
            epoch += 1
            for batch in dataloader:
                if step >= args.max_iters:
                    break

                # Update monitors
                monitor.set_step(step)
                scale_val = scaler.get_scale() if scaler else 1.0
                monitor.set_loss_scale(scale_val)
                if scaler:
                    scaler.set_step(step)

                optimizer.zero_grad(set_to_none=True)

                loss_val = float("nan")
                loss_dict = {}
                skip = False

                # ── Forward ───────────────────────────────────────────────
                try:
                    output = forward_model(model, batch, args.condition)
                    loss_total, loss_dict = aggregate_loss(output)
                    loss_val = loss_total.item()

                    if math.isnan(loss_val) or math.isinf(loss_val):
                        print(f"  ⚠️  step={step} loss={loss_val} — skip backward")
                        skip = True

                except Exception as e:
                    print(f"  [ERROR] Forward step={step}: {e}")
                    skip = True

                # ── Backward ──────────────────────────────────────────────
                if not skip:
                    try:
                        if scaler:
                            scaler.scale(loss_total).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss_total.backward()
                            optimizer.step()
                    except Exception as e:
                        print(f"  [ERROR] Backward step={step}: {e}")
                        skip = True

                # ── Log ───────────────────────────────────────────────────
                log_entry = {
                    "step": step, "epoch": epoch,
                    "loss": loss_val, "skip": skip,
                    "loss_scale": scaler.get_scale() if scaler else 1.0,
                    "overflow_events": monitor.get_overflow_count(),
                    **{f"loss_{k}": v for k, v in loss_dict.items()},
                }
                training_log.append(log_entry)

                if step % 10 == 0:
                    elapsed = time.time() - t_start
                    eta = elapsed / max(step, 1) * (args.max_iters - step)
                    print(f"  step={step:4d}/{args.max_iters} | "
                          f"loss={loss_val:.4f} | "
                          f"scale={scaler.get_scale() if scaler else 1.0:.0f} | "
                          f"ov_events={monitor.get_overflow_count()} | "
                          f"ETA={eta/60:.1f}min")

                step += 1

    except KeyboardInterrupt:
        print("\n  Dihentikan oleh user.")

    finally:
        # ── Simpan training log ───────────────────────────────────────────
        if training_log:
            out_csv = log_dir / "training_log.csv"
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
                writer.writeheader()
                writer.writerows(training_log)
            print(f"\n  Training log → {out_csv}")

        monitor.close()
        if scaler:
            scaler.close()

        elapsed_total = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"  SELESAI: {args.condition.upper()}")
        print(f"  Steps   : {step}/{args.max_iters}")
        print(f"  Overflow: {monitor.get_overflow_count()} events")
        print(f"  Durasi  : {elapsed_total/60:.1f} menit")
        print(f"  Log dir : {log_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
