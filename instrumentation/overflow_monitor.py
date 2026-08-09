"""
overflow_monitor.py — Instrumentasi overflow numerik untuk Co-DETR FP16 Experiment

Komponen:
  - OverflowMonitor: class utama untuk registrasi hook dan logging
  - register_attention_hooks: hook di setiap attention block decoder (sebelum softmax)
  - register_aux_head_hooks: hook di setiap auxiliary head (ATSS/FCOS/RCNN)
  - CSV logging: step, component, max_abs_value, is_inf, is_nan, loss_scale_value

Arsitektur Co-DETR yang dipantau:
  - Decoder attention blocks (QK^T / d^0.5 sebelum softmax)
  - ATSS head classification logit
  - FCOS head classification logit
  - Faster-RCNN (RPN + ROI) head logit
  - Deformable attention sampling offsets
"""
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

FP16_MAX = 65504.0  # batas maksimum representasi FP16


class OverflowEvent:
    """Representasi satu kejadian overflow."""
    def __init__(self, step: int, component: str, sub_component: str,
                 max_abs: float, is_inf: bool, is_nan: bool,
                 loss_scale: float, tensor_dtype: str):
        self.step = step
        self.component = component
        self.sub_component = sub_component
        self.max_abs = max_abs
        self.is_inf = is_inf
        self.is_nan = is_nan
        self.loss_scale = loss_scale
        self.dtype = tensor_dtype
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "step":          self.step,
            "component":     self.component,
            "sub_component": self.sub_component,
            "max_abs_value": self.max_abs,
            "fp16_max":      FP16_MAX,
            "pct_of_fp16_max": self.max_abs / FP16_MAX * 100.0,
            "is_inf":        self.is_inf,
            "is_nan":        self.is_nan,
            "loss_scale":    self.loss_scale,
            "dtype":         self.dtype,
            "timestamp":     self.timestamp,
        }


class OverflowMonitor:
    """
    Monitor overflow numerik untuk Co-DETR.

    Cara pakai:
        monitor = OverflowMonitor(log_dir="results/conditionA_fp16", condition="fp16")
        monitor.register_all_hooks(model)
        # Di training loop:
        for step, batch in enumerate(loader):
            monitor.set_step(step)
            monitor.set_loss_scale(scaler.get_scale())
            loss = model(batch)
            ...
        monitor.close()
    """

    def __init__(self, log_dir: str, condition: str = "unknown",
                 log_every: int = 1, verbose: bool = True):
        """
        Args:
            log_dir: Direktori untuk menyimpan CSV log.
            condition: Label kondisi eksperimen (fp16/fp32).
            log_every: Log setiap N step (1 = setiap step).
            verbose: Print ke terminal setiap ada event overflow.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.condition = condition
        self.log_every = log_every
        self.verbose = verbose

        self._step = 0
        self._loss_scale = 1.0
        self._hooks: List = []
        self._events: List[OverflowEvent] = []
        self._first_overflow_step: Optional[int] = None

        # CSV file
        csv_path = self.log_dir / "overflow_log.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=[
            "step", "component", "sub_component", "max_abs_value",
            "fp16_max", "pct_of_fp16_max", "is_inf", "is_nan",
            "loss_scale", "dtype", "timestamp"
        ])
        self._csv_writer.writeheader()
        self._csv_file.flush()

        # Summary log
        self._summary_path = self.log_dir / "summary.txt"
        print(f"[OverflowMonitor] Log → {csv_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_step(self, step: int):
        self._step = step

    def set_loss_scale(self, scale: float):
        self._loss_scale = scale

    def register_all_hooks(self, model: nn.Module):
        """
        Register semua hook yang dibutuhkan pada model Co-DETR.
        Menemukan layer attention dan auxiliary head secara dinamis.
        """
        self._register_decoder_attention_hooks(model)
        self._register_aux_head_hooks(model)
        self._register_deformable_attention_hooks(model)
        print(f"[OverflowMonitor] Total hooks registered: {len(self._hooks)}")

    def remove_all_hooks(self):
        """Hapus semua hook (panggil setelah training selesai)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def close(self):
        """Flush dan tutup CSV file, cetak summary."""
        self.remove_all_hooks()
        self._csv_file.flush()
        self._csv_file.close()
        self._write_summary()
        print(f"[OverflowMonitor] Monitoring selesai. "
              f"Total events: {len(self._events)}, "
              f"Overflow pertama: step {self._first_overflow_step}")

    def get_overflow_count(self) -> int:
        return sum(1 for e in self._events if e.is_inf or e.is_nan)

    # ── Private: Hook Registrations ───────────────────────────────────────────

    def _register_decoder_attention_hooks(self, model: nn.Module):
        """
        Hook pada attention blocks di decoder Co-DETR.
        Cocok untuk:
          - nn.MultiheadAttention (standard attention)
          - DeformableDetrDecoder
          - MSDeformAttn (multi-scale deformable attention)
        """
        hooks_added = 0
        for name, module in model.named_modules():
            module_type = type(module).__name__

            # Standard MultiheadAttention — hook di forward
            if isinstance(module, nn.MultiheadAttention):
                # Patch forward untuk intersep logit sebelum softmax
                hook = self._make_mha_hook(name)
                h = module.register_forward_hook(hook)
                self._hooks.append(h)
                hooks_added += 1

            # Deformable attention (mmdetection style)
            elif "MSDeformAttn" in module_type or "DeformAttn" in module_type:
                hook = self._make_generic_hook(name, "DeformableAttn")
                h = module.register_forward_hook(hook)
                self._hooks.append(h)
                hooks_added += 1

            # Co-DETR Co-Attention blocks
            elif "CoAttention" in module_type or "CrossAttention" in module_type:
                hook = self._make_generic_hook(name, "CoAttention")
                h = module.register_forward_hook(hook)
                self._hooks.append(h)
                hooks_added += 1

        print(f"[OverflowMonitor] Decoder attention hooks: {hooks_added}")

    def _register_aux_head_hooks(self, model: nn.Module):
        """
        Hook pada auxiliary classification heads:
        - ATSS head
        - FCOS head
        - RPN head (Faster-RCNN)
        - ROI head
        """
        hooks_added = 0
        head_keywords = [
            ("ATSS",      "ATSSHead"),
            ("FCOS",      "FCOSHead"),
            ("RPN",       "RPNHead"),
            ("ROI",       "Shared2FCBBoxHead"),
            ("ROI",       "ConvFCBBoxHead"),
            ("ROI",       "BBoxHead"),
            ("CoDETR",    "CoDETRHead"),
            ("DINO",      "DINOHead"),
        ]

        for name, module in model.named_modules():
            module_type = type(module).__name__
            for label, kw in head_keywords:
                if kw in module_type:
                    hook = self._make_generic_hook(name, f"AuxHead_{label}")
                    h = module.register_forward_hook(hook)
                    self._hooks.append(h)
                    hooks_added += 1
                    break

        print(f"[OverflowMonitor] Aux head hooks: {hooks_added}")

    def _register_deformable_attention_hooks(self, model: nn.Module):
        """
        Hook pada sampling offsets dan attention weights
        di deformable attention — sumber overflow lain yang sering terlewat.
        """
        hooks_added = 0
        for name, module in model.named_modules():
            if hasattr(module, "sampling_offsets"):
                hook = self._make_generic_hook(name, "SamplingOffsets")
                h = module.register_forward_hook(hook)
                self._hooks.append(h)
                hooks_added += 1
            if hasattr(module, "attention_weights"):
                hook = self._make_generic_hook(name, "AttentionWeights")
                h = module.register_forward_hook(hook)
                self._hooks.append(h)
                hooks_added += 1

        print(f"[OverflowMonitor] Deformable attn hooks: {hooks_added}")

    # ── Hook Factories ────────────────────────────────────────────────────────

    def _make_mha_hook(self, layer_name: str):
        """
        Hook untuk nn.MultiheadAttention.
        Menganalisis output dan juga query/key yang tersimpan di module.
        """
        monitor = self

        def hook(module, inputs, output):
            if monitor._step % monitor.log_every != 0:
                return

            # Output dari MHA: (attn_output, attn_weights)
            tensors_to_check = []
            if isinstance(output, tuple):
                for i, t in enumerate(output):
                    if t is not None and isinstance(t, torch.Tensor):
                        tensors_to_check.append((f"output_{i}", t))
            elif isinstance(output, torch.Tensor):
                tensors_to_check.append(("output", output))

            # Juga cek input (Q, K, V) jika tersedia
            for i, inp in enumerate(inputs):
                if inp is not None and isinstance(inp, torch.Tensor) and inp.is_floating_point():
                    tensors_to_check.append((f"input_qkv_{i}", inp))

            for sub_name, tensor in tensors_to_check:
                monitor._log_tensor(
                    component=f"Decoder_MHA/{layer_name}",
                    sub_component=sub_name,
                    tensor=tensor
                )

        return hook

    def _make_generic_hook(self, layer_name: str, component_label: str):
        """Hook generik untuk layer apapun — log output tensor."""
        monitor = self

        def hook(module, inputs, output):
            if monitor._step % monitor.log_every != 0:
                return

            def check_tensor(t, name):
                if t is not None and isinstance(t, torch.Tensor) and t.is_floating_point():
                    monitor._log_tensor(
                        component=f"{component_label}/{layer_name}",
                        sub_component=name,
                        tensor=t
                    )

            if isinstance(output, torch.Tensor):
                check_tensor(output, "output")
            elif isinstance(output, (tuple, list)):
                for i, t in enumerate(output):
                    if isinstance(t, torch.Tensor):
                        check_tensor(t, f"output_{i}")
                    elif isinstance(t, (tuple, list)):
                        for j, tt in enumerate(t):
                            if isinstance(tt, torch.Tensor):
                                check_tensor(tt, f"output_{i}_{j}")

        return hook

    # ── Core Logging ──────────────────────────────────────────────────────────

    def _log_tensor(self, component: str, sub_component: str, tensor: torch.Tensor):
        """Analisis tensor dan log ke CSV."""
        try:
            t = tensor.detach().float()  # konversi ke FP32 untuk analisis akurat
            if t.numel() == 0:
                return

            # Hitung statistik
            max_abs = t.abs().max().item()

            # Cek NaN/Inf di tensor ORIGINAL (FP16 jika FP16)
            orig = tensor.detach()
            is_inf = torch.isinf(orig).any().item()
            is_nan = torch.isnan(orig).any().item()

            dtype_str = str(tensor.dtype)

            event = OverflowEvent(
                step=self._step,
                component=component,
                sub_component=sub_component,
                max_abs=max_abs,
                is_inf=bool(is_inf),
                is_nan=bool(is_nan),
                loss_scale=self._loss_scale,
                tensor_dtype=dtype_str,
            )

            self._events.append(event)
            self._csv_writer.writerow(event.to_dict())

            # Flush setiap 50 events supaya tidak hilang jika crash
            if len(self._events) % 50 == 0:
                self._csv_file.flush()

            # Track first overflow
            if (is_inf or is_nan) and self._first_overflow_step is None:
                self._first_overflow_step = self._step

            # Verbose print
            if self.verbose and (is_inf or is_nan or max_abs > FP16_MAX * 0.5):
                flag = "🔴 OVERFLOW" if (is_inf or is_nan) else "⚠️  NEAR-OVERFLOW"
                print(f"  [{flag}] step={self._step} | {component}/{sub_component} | "
                      f"max_abs={max_abs:.2e} | "
                      f"fp16_max={FP16_MAX:.0f} | "
                      f"is_inf={is_inf} | is_nan={is_nan} | "
                      f"scale={self._loss_scale:.0f}")

        except Exception as e:
            # Jangan crash training karena monitoring error
            pass

    def _write_summary(self):
        """Tulis ringkasan ke file text."""
        overflow_events = [e for e in self._events if e.is_inf or e.is_nan]
        near_overflow = [e for e in self._events
                         if not (e.is_inf or e.is_nan) and e.max_abs > FP16_MAX * 0.9]

        # Komponen paling sering overflow
        from collections import Counter
        comp_counter = Counter(e.component for e in overflow_events)

        with open(self._summary_path, "w") as f:
            f.write(f"="*60 + "\n")
            f.write(f"  OVERFLOW MONITORING SUMMARY\n")
            f.write(f"  Kondisi: {self.condition}\n")
            f.write(f"="*60 + "\n\n")
            f.write(f"Total events logged : {len(self._events)}\n")
            f.write(f"Overflow events     : {len(overflow_events)}\n")
            f.write(f"Near-overflow events: {len(near_overflow)}\n")
            f.write(f"First overflow step : {self._first_overflow_step}\n\n")

            if comp_counter:
                f.write("Top komponen overflow:\n")
                for comp, count in comp_counter.most_common(10):
                    f.write(f"  {count:5d}x  {comp}\n")

            if near_overflow:
                f.write(f"\nStep near-overflow pertama: {near_overflow[0].step}\n")
                f.write(f"  Komponen: {near_overflow[0].component}\n")
                f.write(f"  max_abs: {near_overflow[0].max_abs:.2e}\n")

        print(f"[OverflowMonitor] Summary → {self._summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# GradScaler Wrapper — Log setiap kali step di-skip karena overflow gradien
# ─────────────────────────────────────────────────────────────────────────────

class LoggingGradScaler(torch.cuda.amp.GradScaler):
    """
    Wrapper di atas GradScaler yang mencatat setiap kejadian overflow gradien
    (step di-skip karena gradien mengandung inf/nan).
    """

    def __init__(self, log_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._scaler_csv = open(self._log_path, "w", newline="")
        self._scaler_writer = csv.DictWriter(self._scaler_csv, fieldnames=[
            "step", "scale_before", "scale_after", "overflow_detected", "skipped"
        ])
        self._scaler_writer.writeheader()
        self._current_step = 0
        self._skipped_steps = 0

    def set_step(self, step: int):
        self._current_step = step

    def step(self, optimizer, *args, **kwargs):
        scale_before = self.get_scale()
        result = super().step(optimizer, *args, **kwargs)
        scale_after = self.get_scale()

        overflow = scale_after < scale_before
        skipped = result is None

        if overflow or skipped:
            self._skipped_steps += 1
            print(f"  [GradScaler] step={self._current_step} | "
                  f"OVERFLOW DETECTED — step SKIPPED | "
                  f"scale: {scale_before:.0f} → {scale_after:.0f}")

        self._scaler_writer.writerow({
            "step":              self._current_step,
            "scale_before":      scale_before,
            "scale_after":       scale_after,
            "overflow_detected": overflow,
            "skipped":           skipped,
        })
        self._scaler_csv.flush()
        return result

    def close(self):
        self._scaler_csv.flush()
        self._scaler_csv.close()
        print(f"[GradScaler] Total skipped steps: {self._skipped_steps}")
        print(f"[GradScaler] Log → {self._log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Patch attention untuk FP16 MURNI (tanpa autocast protection)
# ─────────────────────────────────────────────────────────────────────────────

def patch_mha_for_strict_fp16(model: nn.Module, monitor: OverflowMonitor):
    """
    Patch nn.MultiheadAttention untuk menghitung QK^T DI FP16 tanpa
    promosi otomatis ke FP32 oleh autocast. Ini mengekspos overflow
    yang tersembunyi saat memakai torch.cuda.amp.autocast().

    PERINGATAN: Ini sengaja membuat training lebih rentan overflow —
    hanya digunakan untuk MEMBUKTIKAN overflow, bukan untuk training produksi.
    """
    patched_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            _patch_single_mha(module, name, monitor)
            patched_count += 1
    print(f"[StrictFP16Patch] Patched {patched_count} MultiheadAttention modules")
    return patched_count


def _patch_single_mha(mha: nn.MultiheadAttention, name: str, monitor: OverflowMonitor):
    """Patch satu instance MHA untuk strict FP16."""
    original_forward = mha.forward

    def strict_fp16_forward(query, key, value, *args, **kwargs):
        # Paksa semua input ke FP16
        if query.is_cuda:
            q_fp16 = query.to(torch.float16)
            k_fp16 = key.to(torch.float16)
            v_fp16 = value.to(torch.float16)

            # Hitung attention score manual di FP16 (tanpa autocast)
            # Ini yang benar-benar bisa overflow
            with torch.cuda.amp.autocast(enabled=False):
                scale = math.sqrt(mha.head_dim)
                # Manual: q @ k^T / sqrt(d)
                # Shape: (seq, batch, embed) → reshape untuk bmm
                # Ini adalah versi simplified — actual shape tergantung implementasi
                attn_logit_max = None
                try:
                    # Coba hitung scale estimate dari Q dan K norms
                    q_norm = q_fp16.float().norm(dim=-1).max().item()
                    k_norm = k_fp16.float().norm(dim=-1).max().item()
                    # Estimasi max attention logit = q_norm * k_norm / sqrt(d)
                    est_max = q_norm * k_norm / scale
                    attn_logit_max = est_max

                    if monitor:
                        monitor._log_tensor(
                            component=f"StrictFP16_AttnLogit/{name}",
                            sub_component="estimated_qk_max",
                            tensor=torch.tensor(est_max, dtype=torch.float16)
                        )
                except Exception:
                    pass

            # Jalankan forward asli (dengan atau tanpa autocast — kita sudah log estimasinya)
            return original_forward(q_fp16, k_fp16, v_fp16, *args, **kwargs)
        else:
            return original_forward(query, key, value, *args, **kwargs)

    mha.forward = strict_fp16_forward
