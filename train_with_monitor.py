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
import torch
import torch.nn as nn

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

# ═══════════════════════════════════════════════════════════════════════════
# MMCV 1.x → 2.x / mmengine Compatibility Bridge
# Co-DETR ditulis untuk mmcv 1.x, sedangkan kita menggunakan mmcv 2.x.
# Bridge ini memetakan simbol yang sudah dipindahkan agar Co-DETR bisa
# diimport tanpa modifikasi source code-nya.
# ═══════════════════════════════════════════════════════════════════════════
import types, mmcv, mmcv.cnn, mmcv.utils

def _get(dotted, attr, default=None):
    """Import dotted.module.attr dengan fallback default."""
    try:
        mod = __import__(dotted, fromlist=[attr])
        return getattr(mod, attr, default)
    except Exception:
        return default

def _ensure_sub(parent_name, child_name, attrs: dict):
    """Pastikan parent_name.child_name ada di sys.modules dengan attrs."""
    full = f"{parent_name}.{child_name}"
    if full not in sys.modules:
        mod = types.ModuleType(full)
        sys.modules[full] = mod
    mod = sys.modules[full]
    for k, v in attrs.items():
        if not hasattr(mod, k):
            setattr(mod, k, v)
    parent = sys.modules.get(parent_name)
    if parent and not hasattr(parent, child_name):
        setattr(parent, child_name, mod)
    return mod

# ── mmengine stubs ────────────────────────────────────────────────────────
Registry      = _get('mmengine.registry', 'Registry')      or _get('mmcv.utils', 'Registry')
build_from_cfg = _get('mmengine.registry', 'build_from_cfg') or (lambda cfg, reg, default_scope=None: reg.build(cfg))
BaseModule    = _get('mmengine.model',    'BaseModule')     or __import__('torch').nn.Module
load_ckpt     = _get('mmengine.runner',   'load_checkpoint') or (lambda *a, **k: None)

# Dummy hook for eval hooks
class DummyHook:
    def __init__(self, *args, **kwargs): pass

class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
    def forward(self, x):
        return x * self.scale

def auto_fp16(*args, **kwargs):
    def deco(func): return func
    return deco

def force_fp32(*args, **kwargs):
    def deco(func): return func
    return deco

# ── mmcv.utils ──────────────────────────────────────────────────────────
_utils_attrs = dict(
    Registry=Registry,
    build_from_cfg=build_from_cfg,
    build_runner_from_cfg=build_from_cfg,
    import_modules_from_strings=lambda *a, **k: None,
    is_str=lambda x: isinstance(x, str),
    iter_cast=lambda obj, dst_type: list(map(dst_type, obj)),
    print_log=lambda *a, **k: None,
    get_logger=lambda name: __import__('logging').getLogger(name),
    mkdir_or_exist=lambda path, **k: os.makedirs(path, exist_ok=True),
    digit_version=lambda x: [int(y) for y in x.split('.') if y.isdigit()],
)
for _k, _v in _utils_attrs.items():
    if not hasattr(mmcv.utils, _k):
        setattr(mmcv.utils, _k, _v)
_ensure_sub('mmcv', 'utils', _utils_attrs)
_ensure_sub('mmcv.utils', 'registry', dict(Registry=Registry, build_from_cfg=build_from_cfg))
_ensure_sub('mmcv.utils', 'logging', dict(print_log=_utils_attrs['print_log'], get_logger=_utils_attrs['get_logger']))

# ── mmcv.runner ──────────────────────────────────────────────────────────
_runner_attrs = dict(
    BaseModule=BaseModule,
    Sequential=__import__('torch').nn.Sequential,
    ModuleList=__import__('torch').nn.ModuleList,
    auto_fp16=auto_fp16,
    force_fp32=force_fp32,
    load_checkpoint=load_ckpt,
    HOOKS=Registry('hook') if Registry else None,
    RUNNERS=Registry('runner') if Registry else None,
    build_runner=build_from_cfg,
    get_dist_info=lambda: (0, 1),
    master_only=lambda f: f,
    DistEvalHook=DummyHook,
    EvalHook=DummyHook,
)
_ensure_sub('mmcv', 'runner', _runner_attrs)
_ensure_sub('mmcv.runner', 'base_module',  dict(BaseModule=BaseModule))
_ensure_sub('mmcv.runner', 'fp16_utils',   dict(auto_fp16=auto_fp16, force_fp32=force_fp32))
_ensure_sub('mmcv.runner', 'checkpoint',   dict(load_checkpoint=load_ckpt))
_ensure_sub('mmcv.runner', 'hooks',        dict(HOOKS=_runner_attrs.get('HOOKS')))
_ensure_sub('mmcv.runner', 'optimizer',    dict())
_ensure_sub('mmcv.runner', 'dist_utils',   dict(master_only=lambda f: f, get_dist_info=lambda: (0, 1)))

# ── mmcv.cnn ─────────────────────────────────────────────────────────────
def _make_registry(name):
    return Registry(name) if Registry else None

_cnn_attrs = {
    'MODELS':            _make_registry('model') if Registry and not hasattr(mmcv.cnn, 'MODELS') else getattr(mmcv.cnn, 'MODELS', None),
    'Scale':             _get('mmcv.cnn', 'Scale') or Scale,
    'Linear':            _get('mmcv.cnn', 'Linear') or nn.Linear,
    'constant_init':     _get('mmcv.cnn', 'constant_init') or (lambda *a, **k: None),
    'xavier_init':       _get('mmcv.cnn', 'xavier_init') or (lambda *a, **k: None),
    'bias_init_with_prob': _get('mmcv.cnn', 'bias_init_with_prob') or (lambda *a, **k: 0.0),
}
for _k, _v in _cnn_attrs.items():
    if _v is not None and not hasattr(mmcv.cnn, _k):
        setattr(mmcv.cnn, _k, _v)

_cnn_registries = {
    'CONV_LAYERS':       _make_registry('conv_layer'),
    'NORM_LAYERS':       _make_registry('norm_layer'),
    'ACTIVATION_LAYERS': _make_registry('activation_layer'),
    'PLUGIN_LAYERS':     _make_registry('plugin_layer'),
    'UPSAMPLE_LAYERS':   _make_registry('upsample_layer'),
    'PADDING_LAYERS':    _make_registry('padding_layer'),
    'FEEDFORWARD_NETWORK': _make_registry('feedforward_network'),
    'ATTENTION':         _make_registry('attention'),
    'LINEAR_LAYERS':     _make_registry('linear_layer'),
}
for _k, _v in _cnn_registries.items():
    if not hasattr(mmcv.cnn, _k):
        setattr(mmcv.cnn, _k, _v)

_ensure_sub('mmcv.cnn', 'bricks', _cnn_registries)
_ensure_sub('mmcv.cnn.bricks', 'registry', dict(
    NORM_LAYERS=_cnn_registries['NORM_LAYERS'],
    TRANSFORMER_LAYER_SEQUENCE=_make_registry('transformer_layer_sequence'),
))
TransformerLayerSequence = _get('mmcv.cnn.bricks.transformer', 'TransformerLayerSequence') or BaseModule
_ensure_sub('mmcv.cnn.bricks', 'transformer', dict(TransformerLayerSequence=TransformerLayerSequence))
_ensure_sub('mmcv.cnn', 'utils',  dict())
_ensure_sub('mmcv.cnn', 'resnet', dict())

# ── mmcv.parallel ────────────────────────────────────────────────────────
_parallel_attrs = dict(
    MMDataParallel=__import__('torch').nn.DataParallel,
    MMDistributedDataParallel=__import__('torch').nn.parallel.DistributedDataParallel,
    collate=__import__('torch').utils.data.dataloader.default_collate,
    scatter=lambda inputs, target_gpus, dim=0: inputs,
    is_module_wrapper=lambda m: False,
)
_ensure_sub('mmcv', 'parallel', _parallel_attrs)
_ensure_sub('mmcv.parallel', 'data_container', dict())
_ensure_sub('mmcv.parallel', 'distributed', _parallel_attrs)

# ── mmcv.fileio ──────────────────────────────────────────────────────────
try:
    import mmcv.fileio
except Exception:
    _ensure_sub('mmcv', 'fileio', dict(load=lambda f, **k: None, dump=lambda o, f, **k: None))

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
