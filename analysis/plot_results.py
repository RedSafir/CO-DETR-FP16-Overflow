"""
plot_results.py — Visualisasi hasil eksperimen overflow FP16 Co-DETR

Menghasilkan:
  1. plots/fig1_max_abs_value_vs_step.png
     - max_abs_value attention logits & aux head vs step
     - Kondisi A (FP16) merah vs B (FP32) biru
     - Garis horizontal merah putus-putus di FP16_MAX = 65504

  2. plots/fig2_loss_scale_vs_step.png
     - Loss scale GradScaler vs step untuk Kondisi A
     - Titik merah di setiap step yang di-skip karena overflow

  3. plots/fig3_component_breakdown.png
     - Bar chart: jumlah overflow per komponen (ATSS/FCOS/Attention/dll)

  4. plots/fig4_overflow_heatmap.png
     - Heatmap: step (x) vs komponen (y) dengan max_abs_value sebagai warna
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless (tidak perlu display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────
FP16_MAX = 65504.0
# ─────────────────────────────────────────────

def load_overflow_log(csv_path: Path) -> list:
    """Load overflow_log.csv ke list of dicts."""
    if not csv_path.exists():
        print(f"  [WARN] File tidak ditemukan: {csv_path}")
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["step"] = int(row["step"])
            row["max_abs_value"] = float(row["max_abs_value"])
            row["is_inf"] = row["is_inf"].lower() in ("true", "1")
            row["is_nan"] = row["is_nan"].lower() in ("true", "1")
            row["loss_scale"] = float(row.get("loss_scale", 1.0))
            rows.append(row)
    return rows


def load_scaler_log(csv_path: Path) -> list:
    """Load gradscaler_log.csv."""
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["step"] = int(row["step"])
            row["scale_before"] = float(row["scale_before"])
            row["scale_after"] = float(row["scale_after"])
            row["overflow_detected"] = row["overflow_detected"].lower() in ("true", "1")
            row["skipped"] = row["skipped"].lower() in ("true", "1")
            rows.append(row)
    return rows


def aggregate_by_step(rows: list, component_filter: str = None) -> tuple:
    """
    Agregasi max_abs_value per step (ambil maksimum dari semua komponen).
    Jika component_filter diberikan, hanya hitung untuk komponen yang mengandung string itu.
    """
    step_max = defaultdict(float)
    step_inf = defaultdict(bool)
    step_nan = defaultdict(bool)

    for row in rows:
        if component_filter and component_filter not in row.get("component", ""):
            continue
        s = row["step"]
        v = row["max_abs_value"]
        step_max[s] = max(step_max[s], v)
        if row["is_inf"]:
            step_inf[s] = True
        if row["is_nan"]:
            step_nan[s] = True

    steps = sorted(step_max.keys())
    values = [step_max[s] for s in steps]
    is_inf = [step_inf[s] for s in steps]
    is_nan = [step_nan[s] for s in steps]
    return steps, values, is_inf, is_nan


def setup_style():
    """Setup matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#c9d1d9",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "text.color":       "#c9d1d9",
        "grid.color":       "#21262d",
        "grid.linewidth":   0.8,
        "figure.dpi":       150,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "legend.fontsize":  10,
        "legend.framealpha": 0.3,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
    })


def fig1_max_abs_vs_step(fp16_rows, fp32_rows, output_dir: Path):
    """Plot 1: max_abs_value vs step untuk semua komponen, A vs B."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Co-DETR: Nilai Absolut Maksimum Tensor vs Training Step\n"
        "(FP16 Kondisi A vs FP32 Kondisi B)",
        fontsize=14, fontweight="bold", y=0.98
    )

    components = {
        "Semua Komponen":       None,
        "Auxiliary Heads saja": "AuxHead",
    }

    colors = {
        "fp16": {"line": "#f85149", "fill": "#f85149", "overflow": "#ff6e6e"},
        "fp32": {"line": "#3fb950", "fill": "#3fb950", "overflow": "#56d364"},
    }

    for ax, (comp_label, comp_filter) in zip(axes, components.items()):
        # ── FP16 ─────────────────────────────────────────────────────────
        steps_a, vals_a, inf_a, nan_a = aggregate_by_step(fp16_rows, comp_filter)

        if steps_a:
            ax.semilogy(steps_a, vals_a,
                        color=colors["fp16"]["line"], linewidth=1.5,
                        label="Kondisi A (FP16)", alpha=0.9, zorder=3)
            ax.fill_between(steps_a, 1e-3, vals_a,
                            color=colors["fp16"]["fill"], alpha=0.1, zorder=2)

            # Tandai titik overflow (inf/nan)
            overflow_steps = [s for s, i, n in zip(steps_a, inf_a, nan_a) if i or n]
            overflow_vals  = [v for v, i, n in zip(vals_a, inf_a, nan_a) if i or n]
            if overflow_steps:
                ax.scatter(overflow_steps, overflow_vals,
                           color=colors["fp16"]["overflow"], s=60,
                           marker="X", zorder=5, label="Overflow FP16 (Inf/NaN)")

        # ── FP32 ─────────────────────────────────────────────────────────
        steps_b, vals_b, inf_b, nan_b = aggregate_by_step(fp32_rows, comp_filter)

        if steps_b:
            ax.semilogy(steps_b, vals_b,
                        color=colors["fp32"]["line"], linewidth=1.5,
                        label="Kondisi B (FP32)", alpha=0.9, zorder=3)
            ax.fill_between(steps_b, 1e-3, vals_b,
                            color=colors["fp32"]["fill"], alpha=0.08, zorder=2)

        # ── Garis FP16 max ───────────────────────────────────────────────
        ax.axhline(FP16_MAX, color="#ff6e6e", linewidth=2.0,
                   linestyle="--", alpha=0.8, zorder=4,
                   label=f"FP16 Max = {FP16_MAX:.0f}")

        # ── Zona danger (75% dari FP16 max) ────────────────────────────
        ax.axhspan(FP16_MAX * 0.75, FP16_MAX * 2,
                   alpha=0.08, color="#f85149", zorder=1,
                   label="Near-overflow zone (>75% FP16 Max)")

        ax.set_title(f"Filter: {comp_label}", fontweight="bold")
        ax.set_ylabel("max|value| (log scale)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left")
        ax.set_ylim(bottom=1e-2)

    axes[-1].set_xlabel("Training Step")
    plt.tight_layout()

    out = output_dir / "fig1_max_abs_value_vs_step.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


def fig2_loss_scale(scaler_rows, output_dir: Path):
    """Plot 2: Loss scale GradScaler vs step."""
    if not scaler_rows:
        print("  [SKIP] Tidak ada data GradScaler.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        "Co-DETR FP16: Loss Scale GradScaler vs Training Step\n"
        "(Penurunan scale = deteksi overflow gradien)",
        fontsize=13, fontweight="bold"
    )

    steps     = [r["step"]         for r in scaler_rows]
    scales    = [r["scale_before"] for r in scaler_rows]
    overflows = [r for r in scaler_rows if r["overflow_detected"]]

    ax.semilogy(steps, scales,
                color="#58a6ff", linewidth=1.5,
                label="Loss Scale", alpha=0.9, zorder=3)

    if overflows:
        ov_steps  = [r["step"]         for r in overflows]
        ov_scales = [r["scale_before"] for r in overflows]
        ax.scatter(ov_steps, ov_scales,
                   color="#f85149", s=80,
                   marker="v", zorder=5,
                   label=f"Overflow terdeteksi ({len(overflows)}x step di-skip)")
        print(f"  [INFO] Total step di-skip oleh GradScaler: {len(overflows)}")
        print(f"  [INFO] Overflow pertama di step: {overflows[0]['step']}")

    ax.axhline(65536, color="#f0e68c", linewidth=1.5,
               linestyle=":", alpha=0.7, label="Initial scale (65536 = 2^16)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss Scale (log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = output_dir / "fig2_loss_scale_vs_step.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


def fig3_component_breakdown(fp16_rows, output_dir: Path):
    """Plot 3: Bar chart overflow count per komponen."""
    from collections import Counter

    overflow_rows = [r for r in fp16_rows if r["is_inf"] or r["is_nan"]]
    if not overflow_rows:
        print("  [SKIP] Tidak ada overflow events untuk breakdown komponen.")
        return

    # Sederhanakan nama komponen
    def simplify(comp):
        if "AuxHead_ATSS" in comp:   return "ATSS Head"
        if "AuxHead_FCOS" in comp:   return "FCOS Head"
        if "AuxHead_RPN"  in comp:   return "RPN Head"
        if "AuxHead_ROI"  in comp:   return "ROI Head"
        if "Decoder_MHA"  in comp:   return "Decoder Attention"
        if "DeformableAttn" in comp:  return "Deformable Attn"
        if "CoAttention"  in comp:   return "Co-Attention"
        if "StrictFP16"   in comp:   return "Attention Logit (strict)"
        return comp.split("/")[0]

    counts = Counter(simplify(r["component"]) for r in overflow_rows)
    labels = list(counts.keys())
    values = [counts[l] for l in labels]

    # Sort
    sorted_pairs = sorted(zip(values, labels), reverse=True)
    values = [p[0] for p in sorted_pairs]
    labels = [p[1] for p in sorted_pairs]

    # Warna gradasi merah
    cmap = plt.cm.get_cmap("YlOrRd", len(labels))
    bar_colors = [cmap(i / max(len(labels) - 1, 1)) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5 + 2)))
    fig.suptitle(
        "Co-DETR FP16: Breakdown Overflow Events per Komponen\n"
        "(Kondisi A — FP16)",
        fontsize=13, fontweight="bold"
    )

    bars = ax.barh(labels, values, color=bar_colors, edgecolor="#30363d")

    # Anotasi nilai
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=10,
                color="#c9d1d9")

    ax.set_xlabel("Jumlah Overflow Events")
    ax.set_title("", pad=0)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    out = output_dir / "fig3_component_breakdown.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


def fig4_combined_summary(fp16_rows, fp32_rows, scaler_rows, output_dir: Path):
    """Plot 4: Dashboard summary 2x2."""
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Co-DETR FP16 Overflow Experiment — Dashboard Summary",
        fontsize=15, fontweight="bold", y=0.99
    )

    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Subplot 1: max_abs semua komponen ─────────────────────────────────
    steps_a, vals_a, _, _ = aggregate_by_step(fp16_rows)
    steps_b, vals_b, _, _ = aggregate_by_step(fp32_rows)

    if steps_a:
        ax1.semilogy(steps_a, vals_a, color="#f85149", linewidth=1.5, label="FP16 (A)")
    if steps_b:
        ax1.semilogy(steps_b, vals_b, color="#3fb950", linewidth=1.5, label="FP32 (B)")
    ax1.axhline(FP16_MAX, color="#ff6e6e", linestyle="--", linewidth=1.5, label="FP16 Max")
    ax1.set_title("Max |value| — Semua Komponen")
    ax1.set_xlabel("Step"); ax1.set_ylabel("max|value|")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # ── Subplot 2: max_abs hanya AuxHead ─────────────────────────────────
    steps_a2, vals_a2, _, _ = aggregate_by_step(fp16_rows, "AuxHead")
    steps_b2, vals_b2, _, _ = aggregate_by_step(fp32_rows, "AuxHead")

    if steps_a2:
        ax2.semilogy(steps_a2, vals_a2, color="#f85149", linewidth=1.5, label="FP16 AuxHead (A)")
    if steps_b2:
        ax2.semilogy(steps_b2, vals_b2, color="#3fb950", linewidth=1.5, label="FP32 AuxHead (B)")
    ax2.axhline(FP16_MAX, color="#ff6e6e", linestyle="--", linewidth=1.5)
    ax2.set_title("Max |value| — Auxiliary Heads")
    ax2.set_xlabel("Step"); ax2.set_ylabel("max|value|")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # ── Subplot 3: Loss scale ─────────────────────────────────────────────
    if scaler_rows:
        sc_steps  = [r["step"]         for r in scaler_rows]
        sc_scales = [r["scale_before"] for r in scaler_rows]
        sc_ovfl   = [r for r in scaler_rows if r["overflow_detected"]]
        ax3.semilogy(sc_steps, sc_scales, color="#58a6ff", linewidth=1.5, label="Scale")
        if sc_ovfl:
            ax3.scatter([r["step"] for r in sc_ovfl],
                        [r["scale_before"] for r in sc_ovfl],
                        color="#f85149", s=50, marker="v",
                        label=f"Skip ({len(sc_ovfl)}x)")
        ax3.set_title("Loss Scale GradScaler (FP16)")
        ax3.set_xlabel("Step"); ax3.set_ylabel("Loss Scale")
        ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Tidak ada data GradScaler\n(FP32 tidak pakai scaler)",
                 ha="center", va="center", transform=ax3.transAxes,
                 color="#8b949e", fontsize=11)
        ax3.set_title("Loss Scale GradScaler")

    # ── Subplot 4: Ringkasan statistik ────────────────────────────────────
    ax4.axis("off")
    fp16_overflows = sum(1 for r in fp16_rows if r["is_inf"] or r["is_nan"])
    fp32_overflows = sum(1 for r in fp32_rows if r["is_inf"] or r["is_nan"])
    first_overflow = next((r["step"] for r in fp16_rows if r["is_inf"] or r["is_nan"]), "—")
    scaler_skips   = sum(1 for r in scaler_rows if r["overflow_detected"])
    fp16_max_seen  = max((r["max_abs_value"] for r in fp16_rows), default=0.0)
    fp32_max_seen  = max((r["max_abs_value"] for r in fp32_rows), default=0.0)

    stats_text = [
        ("Kondisi A (FP16)", ""),
        ("  Total overflow events",     str(fp16_overflows)),
        ("  Max |value| tercatat",      f"{fp16_max_seen:.2e}"),
        ("  Overflow pertama di step",  str(first_overflow)),
        ("  GradScaler skip steps",     str(scaler_skips)),
        ("", ""),
        ("Kondisi B (FP32)", ""),
        ("  Total overflow events",     str(fp32_overflows)),
        ("  Max |value| tercatat",      f"{fp32_max_seen:.2e}"),
        ("", ""),
        ("FP16 Max (referensi)",        f"{FP16_MAX:.0f}"),
        ("FP16 Max tercapai?",
         "✅ YA" if fp16_max_seen >= FP16_MAX else f"⚠️  {fp16_max_seen/FP16_MAX*100:.1f}% dari max"),
    ]

    y = 0.95
    for label, val in stats_text:
        if not label:
            y -= 0.04
            continue
        if not val:
            ax4.text(0.02, y, label, transform=ax4.transAxes,
                     fontsize=11, fontweight="bold", color="#58a6ff")
        else:
            ax4.text(0.02, y, label, transform=ax4.transAxes,
                     fontsize=10, color="#8b949e")
            ax4.text(0.65, y, val, transform=ax4.transAxes,
                     fontsize=10, color="#c9d1d9", fontweight="bold")
        y -= 0.07

    ax4.set_title("Ringkasan Statistik")

    out = output_dir / "fig4_dashboard_summary.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot Co-DETR FP16 overflow results")
    parser.add_argument("--fp16-dir",   required=True)
    parser.add_argument("--fp32-dir",   required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    fp16_dir = Path(args.fp16_dir)
    fp32_dir = Path(args.fp32_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  ANALISIS & VISUALISASI HASIL EKSPERIMEN")
    print("="*60)

    print("\n[1/5] Load data overflow logs...")
    fp16_rows   = load_overflow_log(fp16_dir / "overflow_log.csv")
    fp32_rows   = load_overflow_log(fp32_dir / "overflow_log.csv")
    scaler_rows = load_scaler_log(fp16_dir / "gradscaler_log.csv")

    print(f"  FP16 events : {len(fp16_rows)}")
    print(f"  FP32 events : {len(fp32_rows)}")
    print(f"  Scaler rows : {len(scaler_rows)}")

    setup_style()

    print("\n[2/5] Plot 1: max_abs_value vs step...")
    fig1_max_abs_vs_step(fp16_rows, fp32_rows, out_dir)

    print("[3/5] Plot 2: Loss scale vs step...")
    fig2_loss_scale(scaler_rows, out_dir)

    print("[4/5] Plot 3: Komponen breakdown...")
    fig3_component_breakdown(fp16_rows, out_dir)

    print("[5/5] Plot 4: Dashboard summary...")
    fig4_combined_summary(fp16_rows, fp32_rows, scaler_rows, out_dir)

    print(f"\n{'='*60}")
    print(f"  PLOT SELESAI → {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
