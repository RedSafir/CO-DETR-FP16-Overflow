#!/usr/bin/env python3
"""
train_standalone.py - Co-DETR FP16 Overflow Experiment (Standalone, no mmcv/mmdet)
Arsitektur: FPN + 6-layer Transformer Decoder + ATSS + FCOS + RPN heads
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

FP16_MAX = 65504.0


# ── Arsitektur ─────────────────────────────────────────────────────────────────

class FPN(nn.Module):
    def __init__(self, d=256, n=5):
        super().__init__()
        self.lats = nn.ModuleList([nn.Conv2d(d, d, 1) for _ in range(n)])
        self.outs = nn.ModuleList([nn.Conv2d(d, d, 3, padding=1) for _ in range(n)])

    def forward(self, x):
        h, w = x.shape[-2:]
        return [
            self.outs[i](self.lats[i](
                F.interpolate(x, (max(1, h // 2**i), max(1, w // 2**i)),
                              mode='bilinear', align_corners=False)
            ))
            for i in range(len(self.lats))
        ]


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention approx - sumber overflow utama FP16."""
    def __init__(self, d=256, h=8, lvl=5, pts=4, large_init=False):
        super().__init__()
        self.h, self.hd = h, d // h
        self.large_init = large_init
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.offsets = nn.Linear(d, h * lvl * pts * 2)
        self.weights = nn.Linear(d, h * lvl * pts)
        self._reset_parameters()

    def _reset_parameters(self):
        if self.large_init:
            # std=1.0 mensimulasikan magnitude weight Co-DETR setelah pre-training.
            # Dengan LN output std=1 dan weight std=1:
            #   Q, K output std = sqrt(d_model) * std_w = sqrt(256) * 1 = 16
            #   Logit Q@K^T / sqrt(d_k) std = 16 * 16 = 256  >> threshold overflow softmax FP16 (~88)
            nn.init.normal_(self.q.weight, std=1.0)
            nn.init.normal_(self.k.weight, std=1.0)
            nn.init.normal_(self.v.weight, std=0.5)
            nn.init.normal_(self.o.weight, std=0.5)
        else:
            nn.init.xavier_uniform_(self.q.weight)
            nn.init.xavier_uniform_(self.k.weight)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    def forward(self, q, kv=None):
        if kv is None:
            kv = q
        B, N, C = q.shape
        Q = self.q(q).view(B, N, self.h, self.hd).transpose(1, 2)
        K = self.k(kv).view(B, -1, self.h, self.hd).transpose(1, 2)
        V = self.v(kv).view(B, -1, self.h, self.hd).transpose(1, 2)
        # Attention logits — TITIK OVERFLOW UTAMA di FP16
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hd)
        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, N, C)
        return self.o(out), logits, self.offsets(q), self.weights(q)


class DecLayer(nn.Module):
    def __init__(self, d=256, h=8, ff=2048, drop=0.1, large_init=False):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ca = MSDeformAttn(d, h, large_init=large_init)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, q, mem):
        q2, _ = self.sa(q, q, q)
        q = self.n1(q + self.drop(q2))
        q3, logits, offs, w = self.ca(q, mem)
        q = self.n2(q + self.drop(q3))
        q = self.n3(q + self.drop(self.ff(q)))
        return q, logits, offs


class ATSSHead(nn.Module):
    def __init__(self, d=256, nc=80, sc=4):
        super().__init__()
        self.cls_c = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.ReLU()) for _ in range(sc)]
        )
        self.reg_c = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.ReLU()) for _ in range(sc)]
        )
        self.cls = nn.Conv2d(d, nc, 1)
        self.reg = nn.Conv2d(d, 4, 1)
        self.ctr = nn.Conv2d(d, 1, 1)

    def forward(self, feats):
        return (
            [self.cls(self.cls_c(f)) for f in feats],
            [self.reg(self.reg_c(f)) for f in feats],
            [self.ctr(self.cls_c(f)) for f in feats],
        )


class FCOSHead(nn.Module):
    def __init__(self, d=256, nc=80):
        super().__init__()
        self.ct = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU(),
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU(),
        )
        self.cl = nn.Conv2d(d, nc, 1)
        self.rg = nn.Conv2d(d, 4, 1)

    def forward(self, feats):
        return [self.cl(self.ct(f)) for f in feats], [self.rg(f) for f in feats]


class RPNHead(nn.Module):
    def __init__(self, d=256, na=3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.ReLU())
        self.cl = nn.Conv2d(d, na, 1)
        self.rg = nn.Conv2d(d, na * 4, 1)

    def forward(self, feats):
        return (
            [self.cl(self.conv(f)) for f in feats],
            [self.rg(self.conv(f)) for f in feats],
        )


class QueryHead(nn.Module):
    def __init__(self, d=256, nc=80, nl=6):
        super().__init__()
        self.cls = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, nc)) for _ in range(nl)
        ])
        self.reg = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 4)) for _ in range(nl)
        ])

    def forward(self, outs):
        return (
            [self.cls[i](o) for i, o in enumerate(outs)],
            [self.reg[i](o).sigmoid() for i, o in enumerate(outs)],
        )


class CoDETR(nn.Module):
    """Replika arsitektur Co-DETR dalam PyTorch murni (tanpa mmcv/mmdet)."""
    def __init__(self, d=256, nc=80, nq=300, nl=6, large_init=False):
        super().__init__()
        self.proj  = nn.Conv2d(3, d, 7, stride=4, padding=3)
        self.fpn   = FPN(d)
        self.qemb  = nn.Embedding(nq, d)
        self.dec   = nn.ModuleList([DecLayer(d, large_init=large_init) for _ in range(nl)])
        self.qhead = QueryHead(d, nc, nl)
        self.atss  = ATSSHead(d, nc)
        self.fcos  = FCOSHead(d, nc)
        self.rpn   = RPNHead(d)

    def forward(self, x):
        B = x.shape[0]
        feats = self.fpn(self.proj(x))
        mem = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], dim=1)
        q = self.qemb.weight.unsqueeze(0).expand(B, -1, -1)
        dec_outs, all_logits, all_offs = [], [], []
        for layer in self.dec:
            q, logits, offs = layer(q, mem)
            dec_outs.append(q)
            all_logits.append(logits)
            all_offs.append(offs)
        mc, mr = self.qhead(dec_outs)
        ac, ar, actr = self.atss(feats)
        fc, fr = self.fcos(feats)
        rc, rr = self.rpn(feats)
        return dict(
            main_cls=mc, main_reg=mr,
            atss_cls=ac, atss_reg=ar, atss_ctr=actr,
            fcos_cls=fc, fcos_reg=fr,
            rpn_cls=rc, rpn_reg=rr,
            attn_logits=all_logits, offsets=all_offs,
        )


# ── Dataset ────────────────────────────────────────────────────────────────────

class RandDet(Dataset):
    def __init__(self, n=600):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.randn(3, 640, 640), torch.rand(8, 4), torch.randint(0, 80, (8,))


def collate(b):
    imgs, bx, lb = zip(*b)
    return torch.stack(imgs), list(bx), list(lb)


# ── Loss ───────────────────────────────────────────────────────────────────────

def loss_fn(out, dev):
    t = torch.tensor(0., device=dev, requires_grad=True)
    for c, r in zip(out['main_cls'], out['main_reg']):
        t = t + F.binary_cross_entropy_with_logits(c, torch.zeros_like(c))
        t = t + F.l1_loss(r, torch.rand_like(r))
    for c, r, _ in zip(out['atss_cls'], out['atss_reg'], out['atss_ctr']):
        t = t + F.binary_cross_entropy_with_logits(c, torch.zeros_like(c))
    for c, _ in zip(out['fcos_cls'], out['fcos_reg']):
        t = t + F.binary_cross_entropy_with_logits(c, torch.zeros_like(c))
    for c, _ in zip(out['rpn_cls'], out['rpn_reg']):
        t = t + F.binary_cross_entropy_with_logits(c, torch.zeros_like(c))
    return t


# ── Monitor ────────────────────────────────────────────────────────────────────

class Monitor:
    def __init__(self, log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._f = open(Path(log_dir) / 'overflow_log.csv', 'w', newline='')
        self._w = csv.DictWriter(self._f, [
            'step', 'component', 'sub', 'max_abs',
            'pct_fp16max', 'is_inf', 'is_nan', 'scale', 'dtype', 'ts'
        ])
        self._w.writeheader()
        self.events = []

    def log(self, step, comp, sub, t, scale):
        if not t.is_floating_point():
            return
        fp = t.detach().float()
        mx = fp.abs().max().item()
        ii = bool(torch.isinf(t.detach()).any())
        in_ = bool(torch.isnan(t.detach()).any())
        pct = mx / FP16_MAX * 100
        row = dict(step=step, component=comp, sub=sub, max_abs=f'{mx:.4e}',
                   pct_fp16max=f'{pct:.2f}', is_inf=ii, is_nan=in_,
                   scale=scale, dtype=str(t.dtype), ts=time.strftime('%H:%M:%S'))
        self._w.writerow(row)
        self.events.append(row)
        if len(self.events) % 200 == 0:
            self._f.flush()
        if ii or in_ or pct > 50:
            tag = "OVERFLOW" if (ii or in_) else "NEAR-OVF"
            print(f"  [{tag}] step={step:4d} {comp}/{sub} "
                  f"max={mx:.2e} ({pct:.1f}%) inf={ii} nan={in_}")

    def log_outputs(self, step, out, scale):
        for i, lg in enumerate(out.get('attn_logits', [])):
            self.log(step, 'CrossAttn', f'l{i}', lg, scale)
        for i, of in enumerate(out.get('offsets', [])):
            self.log(step, 'Offsets', f'l{i}', of, scale)
        for i, c in enumerate(out.get('atss_cls', [])):
            self.log(step, 'ATSS', f'cls{i}', c, scale)
        for i, c in enumerate(out.get('fcos_cls', [])):
            self.log(step, 'FCOS', f'cls{i}', c, scale)
        for i, c in enumerate(out.get('rpn_cls', [])):
            self.log(step, 'RPN', f'cls{i}', c, scale)
        for i, c in enumerate(out.get('main_cls', [])):
            self.log(step, 'MainHead', f'cls{i}', c, scale)

    def close(self):
        self._f.flush()
        self._f.close()
        n_ov   = sum(1 for e in self.events if e['is_inf'] or e['is_nan'])
        n_near = sum(1 for e in self.events if float(e['pct_fp16max']) > 50)
        print(f"\n[Monitor] Events={len(self.events)} | Overflow={n_ov} | Near-ovf={n_near}")


class LogScaler(torch.cuda.amp.GradScaler):
    def __init__(self, path, **kw):
        super().__init__(**kw)
        self._step = 0
        self._f = open(path, 'w', newline='')
        self._w = csv.DictWriter(self._f, ['step', 'scale', 'overflow', 'ts'])
        self._w.writeheader()

    def update(self):
        prev = self.get_scale()
        super().update()
        new = self.get_scale()
        ov = new < prev
        self._w.writerow(dict(step=self._step, scale=new, overflow=ov,
                              ts=time.strftime('%H:%M:%S')))
        self._f.flush()
        return ov

    def close(self):
        self._f.flush()
        self._f.close()


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(cond, iters, log_dir, bs, seed, force_overflow=False):
    torch.manual_seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'N/A'
    print(f"\n  Device: {dev} ({gpu_name})")
    print(f"  Mode  : {cond.upper()} | force_overflow={force_overflow}")
    if force_overflow:
        print("  [!] Large-init mode aktif — mensimulasikan weight Co-DETR setelah pre-training")
        print("      Q,K weight std=1.0 → logit std ≈ 256 >> FP16 softmax overflow threshold (88)")

    model = CoDETR(large_init=force_overflow).to(dev)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {n_params:.1f}M")

    loader = DataLoader(
        RandDet(max(600, iters * bs)), bs, shuffle=True,
        num_workers=2, collate_fn=collate, pin_memory=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    mon = Monitor(log_dir)

    scaler = None
    if cond == 'fp16':
        scaler = LogScaler(
            str(Path(log_dir) / 'gradscaler_log.csv'),
            init_scale=2**16, growth_factor=2.0,
            backoff_factor=0.5, growth_interval=2000,
        )

    tlog = []
    step = 0
    t0 = time.time()
    print(f"\n  Training {cond.upper()} — {iters} steps...\n")
    model.train()

    try:
        while step < iters:
            for batch in loader:
                if step >= iters:
                    break
                imgs = batch[0].to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                if cond == 'fp16':
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = model(imgs)
                        loss = loss_fn(out, dev)
                else:
                    out = model(imgs)
                    loss = loss_fn(out, dev)

                lv = loss.item()
                sc = scaler.get_scale() if scaler else 1.0
                mon.log_outputs(step, out, sc)

                if not (math.isnan(lv) or math.isinf(lv)):
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                        scaler._step = step
                    else:
                        loss.backward()
                        opt.step()

                tlog.append({'step': step, 'loss': lv, 'scale': sc,
                             'time': time.time() - t0})

                if step % 10 == 0:
                    el = time.time() - t0
                    eta = el / max(step, 1) * (iters - step)
                    print(f"  step={step:4d}/{iters} loss={lv:.4f} "
                          f"scale={sc:.0f} elapsed={el:.0f}s eta={eta:.0f}s")
                step += 1

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user.")

    with open(Path(log_dir) / 'training_log.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['step', 'loss', 'scale', 'time'])
        w.writeheader()
        w.writerows(tlog)

    mon.close()
    if scaler:
        scaler.close()
    print(f"\n  DONE {cond.upper()} — {step} steps in {time.time()-t0:.1f}s")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Co-DETR FP16 Overflow Experiment (Standalone)")
    p.add_argument('--condition',       required=True, choices=['fp16', 'fp32'])
    p.add_argument('--max-iters',       type=int, default=300)
    p.add_argument('--log-dir',         required=True)
    p.add_argument('--batch-size',      type=int, default=2)
    p.add_argument('--seed',            type=int, default=42)
    p.add_argument('--force-overflow',  action='store_true',
                   help='Gunakan large weight init (std=1.0) untuk Q,K '
                        'mensimulasikan trained Co-DETR weights dan memaksa FP16 overflow')
    a = p.parse_args()
    train(a.condition, a.max_iters, a.log_dir, a.batch_size, a.seed,
          force_overflow=a.force_overflow)


if __name__ == '__main__':
    main()
