"""
generate_report.py — Generate laporan Markdown dari hasil eksperimen overflow

Output: REPORT.md — ringkasan temuan overflow FP16 Co-DETR
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime


FP16_MAX = 65504.0


def load_overflow_log(path):
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["step"] = int(row["step"])
            row["max_abs_value"] = float(row["max_abs_value"])
            row["is_inf"] = row["is_inf"].lower() in ("true", "1")
            row["is_nan"] = row["is_nan"].lower() in ("true", "1")
            row["loss_scale"] = float(row.get("loss_scale", 1.0))
            rows.append(row)
    return rows


def load_scaler_log(path):
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["step"] = int(row["step"])
            row["scale_before"] = float(row["scale_before"])
            row["scale_after"] = float(row["scale_after"])
            row["overflow_detected"] = row["overflow_detected"].lower() in ("true", "1")
            rows.append(row)
    return rows


def analyze(rows, scaler_rows=None):
    """Analisis mendalam dari overflow log."""
    total = len(rows)
    overflow = [r for r in rows if r["is_inf"] or r["is_nan"]]
    near_overflow = [r for r in rows if not (r["is_inf"] or r["is_nan"])
                     and r["max_abs_value"] > FP16_MAX * 0.5]

    first_overflow_step = overflow[0]["step"] if overflow else None
    first_overflow_comp = overflow[0]["component"] if overflow else None

    max_abs_seen = max((r["max_abs_value"] for r in rows), default=0.0)
    max_abs_step = next((r["step"] for r in rows
                         if r["max_abs_value"] == max_abs_seen), None)

    # Komponen paling overflow
    def simplify(comp):
        if "AuxHead_ATSS" in comp: return "ATSS Head"
        if "AuxHead_FCOS" in comp: return "FCOS Head"
        if "AuxHead_RPN"  in comp: return "RPN Head"
        if "AuxHead_ROI"  in comp: return "ROI/BBox Head"
        if "Decoder_MHA"  in comp: return "Decoder Attention (MHA)"
        if "DeformableAttn" in comp: return "Deformable Attention"
        if "CoAttention"  in comp: return "Co-Attention"
        return comp.split("/")[0]

    comp_overflow = Counter(simplify(r["component"]) for r in overflow)
    comp_near     = Counter(simplify(r["component"]) for r in near_overflow)

    # Scaler analysis
    scaler_skips = 0
    first_scaler_skip = None
    if scaler_rows:
        skips = [r for r in scaler_rows if r["overflow_detected"]]
        scaler_skips = len(skips)
        first_scaler_skip = skips[0]["step"] if skips else None

    return {
        "total_events":        total,
        "overflow_count":      len(overflow),
        "near_overflow_count": len(near_overflow),
        "first_overflow_step": first_overflow_step,
        "first_overflow_comp": first_overflow_comp,
        "max_abs_seen":        max_abs_seen,
        "max_abs_step":        max_abs_step,
        "comp_overflow":       comp_overflow,
        "comp_near":           comp_near,
        "scaler_skips":        scaler_skips,
        "first_scaler_skip":   first_scaler_skip,
    }


def generate_markdown(fp16_dir, fp32_dir, out_path):
    fp16_rows   = load_overflow_log(fp16_dir / "overflow_log.csv")
    fp32_rows   = load_overflow_log(fp32_dir / "overflow_log.csv")
    scaler_rows = load_scaler_log(fp16_dir / "gradscaler_log.csv")

    a = analyze(fp16_rows, scaler_rows)
    b = analyze(fp32_rows)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    overflow_confirmed = a["overflow_count"] > 0
    near_confirmed     = a["near_overflow_count"] > 0
    fp32_clean         = b["overflow_count"] == 0

    # Tentukan komponen paling rentan
    most_vulnerable = a["comp_overflow"].most_common(1)
    most_vulnerable_name = most_vulnerable[0][0] if most_vulnerable else "N/A"

    # Jenis overflow
    has_activation_overflow = a["first_overflow_step"] is not None  # dari forward hook
    has_gradient_overflow   = a["scaler_skips"] > 0                 # dari GradScaler

    lines = []
    lines.append(f"# Laporan Eksperimen: Pembuktian Overflow FP16 pada Co-DETR")
    lines.append(f"")
    lines.append(f"**Tanggal:** {now}")
    lines.append(f"**Hardware:** NVIDIA GeForce RTX 5060 Ti (sm_120, 16GB)")
    lines.append(f"**Model:** Co-DETR dengan backbone ResNet-50")
    lines.append(f"**Dataset:** COCO val2017 subset (~500 gambar)")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # ── Executive Summary ──────────────────────────────────────────────────
    lines.append(f"## 1. Ringkasan Eksekutif")
    lines.append(f"")

    verdict_a = "✅ **TERBUKTI**" if overflow_confirmed else "⚠️ Near-overflow terdeteksi, inf/nan belum terpicu"
    verdict_b = "✅ **BERSIH**" if fp32_clean else "⚠️ Overflow juga terdeteksi di FP32 (tidak terduga)"

    lines.append(f"| Kondisi | Status Overflow | Total Events | Max |value| |")
    lines.append(f"|---------|----------------|--------------|-------------|")
    lines.append(f"| A (FP16 + dynamic scale) | {verdict_a} | {a['overflow_count']} | {a['max_abs_seen']:.4e} |")
    lines.append(f"| B (FP32 baseline)        | {verdict_b}          | {b['overflow_count']} | {b['max_abs_seen']:.4e} |")
    lines.append(f"")

    if overflow_confirmed and fp32_clean:
        lines.append(f"> **Kesimpulan utama:** Overflow FP16 TERKONFIRMASI pada Co-DETR.")
        lines.append(f"> Kondisi A (FP16) mengalami {a['overflow_count']} overflow events,")
        lines.append(f"> sementara Kondisi B (FP32) tetap bersih di semua {b['total_events']} measurements.")
    elif near_confirmed:
        lines.append(f"> **Kesimpulan utama:** Near-overflow terdeteksi (max |value| mencapai")
        lines.append(f"> {a['max_abs_seen']:.2e} dari batas FP16 {FP16_MAX:.0f}).")
        lines.append(f"> Eksperimen lebih panjang atau batch lebih besar kemungkinan akan memicu overflow penuh.")
    else:
        lines.append(f"> **Catatan:** Overflow belum terdeteksi dalam {a['total_events']} measurements.")
        lines.append(f"> Pertimbangkan menjalankan lebih banyak iterasi atau batch size lebih besar.")

    lines.append(f"")

    # ── Temuan Per Komponen ────────────────────────────────────────────────
    lines.append(f"## 2. Temuan Per Komponen")
    lines.append(f"")
    lines.append(f"### 2.1 Komponen Paling Rentan (Kondisi A — FP16)")
    lines.append(f"")

    if a["comp_overflow"]:
        lines.append(f"| Komponen | Jumlah Overflow Events | Persentase |")
        lines.append(f"|----------|----------------------|------------|")
        total_ov = sum(a["comp_overflow"].values())
        for comp, cnt in a["comp_overflow"].most_common():
            pct = cnt / total_ov * 100
            lines.append(f"| {comp} | {cnt} | {pct:.1f}% |")
        lines.append(f"")
        lines.append(f"**Komponen paling rentan:** `{most_vulnerable_name}`")
        lines.append(f"")
    else:
        lines.append(f"*Tidak ada overflow events untuk dibreakdown.*")
        lines.append(f"")

    lines.append(f"### 2.2 Near-Overflow Events (>50% dari FP16 Max)")
    lines.append(f"")
    if a["comp_near"]:
        lines.append(f"| Komponen | Near-Overflow Events |")
        lines.append(f"|----------|---------------------|")
        for comp, cnt in a["comp_near"].most_common():
            lines.append(f"| {comp} | {cnt} |")
        lines.append(f"")
    else:
        lines.append(f"*Tidak ada near-overflow events.*")
        lines.append(f"")

    # ── Jenis Overflow ─────────────────────────────────────────────────────
    lines.append(f"## 3. Jenis Overflow yang Terdeteksi")
    lines.append(f"")
    lines.append(f"| Jenis | Mekanisme | Terdeteksi? | Detail |")
    lines.append(f"|-------|-----------|-------------|--------|")
    lines.append(f"| **Overflow Aktivasi** (forward pass) | Forward hook pada attention & aux heads | "
                 f"{'✅ Ya' if has_activation_overflow else '❌ Tidak'} | "
                 f"Step pertama: {a['first_overflow_step'] or 'N/A'} ({a['first_overflow_comp'] or 'N/A'}) |")
    lines.append(f"| **Overflow Gradien** (backward pass) | GradScaler deteksi inf/nan di gradien | "
                 f"{'✅ Ya' if has_gradient_overflow else '❌ Tidak'} | "
                 f"Step pertama: {a['first_scaler_skip'] or 'N/A'}, total skip: {a['scaler_skips']} |")
    lines.append(f"")

    if has_activation_overflow:
        lines.append(f"> **Penting:** Overflow aktivasi (forward) **TIDAK** bisa dicegah oleh GradScaler.")
        lines.append(f"> Ini berarti NaN/Inf sudah terjadi sebelum backward pass, sehingga loss menjadi")
        lines.append(f"> NaN terlebih dahulu — GradScaler kemudian mendeteksi NaN di gradien sebagai efek sekunder.")
        lines.append(f"")

    # ── Timeline ──────────────────────────────────────────────────────────
    lines.append(f"## 4. Timeline Overflow")
    lines.append(f"")
    lines.append(f"| Kejadian | Step | Kondisi |")
    lines.append(f"|----------|------|---------|")
    if a["first_overflow_step"] is not None:
        lines.append(f"| Overflow pertama (forward hook) | {a['first_overflow_step']} | A (FP16) |")
    if a["first_scaler_skip"] is not None:
        lines.append(f"| Overflow pertama (GradScaler) | {a['first_scaler_skip']} | A (FP16) |")
    lines.append(f"| Max |value| = {a['max_abs_seen']:.2e} | {a['max_abs_step']} | A (FP16) |")
    if b["first_overflow_step"] is not None:
        lines.append(f"| Overflow pertama | {b['first_overflow_step']} | B (FP32) — tidak terduga! |")
    else:
        lines.append(f"| Tidak ada overflow | — | B (FP32) |")
    lines.append(f"")

    # ── GradScaler Log ─────────────────────────────────────────────────────
    lines.append(f"## 5. Log GradScaler (Kondisi A — FP16)")
    lines.append(f"")
    lines.append(f"- Total steps yang di-skip karena overflow gradien: **{a['scaler_skips']}**")
    if a["first_scaler_skip"]:
        lines.append(f"- Pertama kali skip di step: **{a['first_scaler_skip']}**")
    lines.append(f"- Log detail tersimpan di: `results/conditionA_fp16/gradscaler_log.csv`")
    lines.append(f"")

    # ── Rekomendasi ────────────────────────────────────────────────────────
    lines.append(f"## 6. Rekomendasi Mitigasi")
    lines.append(f"")
    lines.append(f"Berdasarkan bukti eksperimen, berikut adalah rekomendasi mitigasi dari yang paling efektif:")
    lines.append(f"")

    recs = [
        ("BF16 (Brain Float 16)", "HIGH",
         "Ganti FP16 dengan BF16. Range BF16 identik dengan FP32 (~3.4e38 vs 65504) sehingga overflow tidak mungkin terjadi. "
         "RTX 5060 Ti (sm_120) mendukung BF16 native. `autocast(dtype=torch.bfloat16)`",
         "`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True`"),
        ("QK-Normalization di Decoder", "HIGH",
         "Tambahkan LayerNorm pada Q dan K sebelum dot-product attention. "
         "Mencegah attention logit meledak saat query/key memiliki magnitude besar. "
         "Diimplementasikan di ViT-22B dan model DETR terbaru.",
         "`nn.LayerNorm(d_model)` di TransformerDecoder sebelum `q @ k.T`"),
        ("TF32 (TensorFloat-32)", "MEDIUM",
         "Aktifkan TF32 untuk operasi matmul — memakai akumulator FP32 internal tapi tetap cepat. "
         "Secara teknis masih FP32 range, sehingga tidak overflow.",
         "`torch.backends.cuda.matmul.allow_tf32 = True`"),
        ("Per-head Loss Weighting", "MEDIUM",
         "Berikan weight berbeda untuk loss dari setiap auxiliary head. "
         "ATSS/FCOS loss scale bisa berbeda order of magnitude dengan head utama, "
         "menyebabkan loss gabungan memiliki magnitudo yang sulit diskalakan dengan satu GradScaler.",
         "Tambahkan `loss_weight` per auxiliary head di config Co-DETR"),
        ("Gradient Clipping Lebih Agresif", "LOW",
         "Kurangi `max_norm` di `grad_clip` dari 0.1 ke 0.01. "
         "Mencegah gradien besar memperbesar parameter hingga aktivasi overflow di iterasi berikutnya.",
         "`optimizer_config = dict(grad_clip=dict(max_norm=0.01))`"),
    ]

    lines.append(f"| # | Solusi | Prioritas | Cara Implementasi |")
    lines.append(f"|---|--------|-----------|------------------|")
    for i, (name, priority, desc, impl) in enumerate(recs, 1):
        priority_icon = "🔴" if priority == "HIGH" else ("🟡" if priority == "MEDIUM" else "🟢")
        lines.append(f"| {i} | **{name}** | {priority_icon} {priority} | {impl} |")
    lines.append(f"")

    for i, (name, priority, desc, impl) in enumerate(recs, 1):
        lines.append(f"### 6.{i} {name}")
        lines.append(f"")
        lines.append(f"{desc}")
        lines.append(f"")
        lines.append(f"```python")
        lines.append(impl)
        lines.append(f"```")
        lines.append(f"")

    # ── File Output ───────────────────────────────────────────────────────
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Lampiran: File Output")
    lines.append(f"")
    lines.append(f"| File | Deskripsi |")
    lines.append(f"|------|-----------|")
    lines.append(f"| `results/conditionA_fp16/overflow_log.csv` | Log detail setiap tensor yang dimonitor (FP16) |")
    lines.append(f"| `results/conditionA_fp16/gradscaler_log.csv` | Log GradScaler per step — kapan scale turun |")
    lines.append(f"| `results/conditionA_fp16/training_log.csv` | Loss per step (FP16) |")
    lines.append(f"| `results/conditionB_fp32/overflow_log.csv` | Log detail setiap tensor yang dimonitor (FP32) |")
    lines.append(f"| `results/plots/fig1_max_abs_value_vs_step.png` | Plot max |value| vs step — A vs B |")
    lines.append(f"| `results/plots/fig2_loss_scale_vs_step.png` | Plot loss scale GradScaler vs step |")
    lines.append(f"| `results/plots/fig3_component_breakdown.png` | Bar chart overflow per komponen |")
    lines.append(f"| `results/plots/fig4_dashboard_summary.png` | Dashboard ringkasan 4-panel |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"*Laporan ini di-generate otomatis oleh `analysis/generate_report.py`*")

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    print(f"  [OK] Laporan → {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-dir", required=True)
    parser.add_argument("--fp32-dir", required=True)
    parser.add_argument("--output",   default="REPORT.md")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  GENERATE LAPORAN AKHIR")
    print("="*60)

    report = generate_markdown(
        fp16_dir=Path(args.fp16_dir),
        fp32_dir=Path(args.fp32_dir),
        out_path=Path(args.output),
    )

    print(f"\nPreview (baris pertama):")
    for line in report.split("\n")[:15]:
        print(f"  {line}")
    print(f"  ... (total {len(report.split(chr(10)))} baris)")


if __name__ == "__main__":
    main()
