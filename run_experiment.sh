#!/bin/bash
# =============================================================================
# run_experiment.sh — Jalankan semua kondisi eksperimen Co-DETR FP16 Overflow
# =============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MAX_ITERS=300
BATCH_SIZE=2
DATA_ROOT="data/coco_subset"
RESULTS_DIR="results"
CONDA_ENV="codetr_fp16_exp"

# ─────────────────────────────────────────────
# Aktifkan conda environment
# ─────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV" 2>/dev/null || log_warn "Tidak bisa aktifkan conda env '$CONDA_ENV', lanjut dengan env saat ini."

# ─────────────────────────────────────────────
# Verifikasi environment
# ─────────────────────────────────────────────
log_step "0. Verifikasi Environment"
python setup/verify_env.py || {
    log_error "Verifikasi environment gagal! Jalankan setup/install.sh terlebih dahulu."
    exit 1
}

# ─────────────────────────────────────────────
# Cek dataset
# ─────────────────────────────────────────────
log_step "1. Cek Dataset"
if [ ! -f "$DATA_ROOT/annotations/instances_val2017_subset.json" ]; then
    log_warn "Dataset belum ada. Mengunduh..."
    python data/prepare_dataset.py --output-dir "$DATA_ROOT" --n-images 500
else
    N_IMGS=$(python -c "
import json
with open('$DATA_ROOT/annotations/instances_val2017_subset.json') as f:
    d = json.load(f)
print(len(d['images']))
" 2>/dev/null || echo "?")
    log_info "Dataset sudah ada: $N_IMGS gambar"
fi

# ─────────────────────────────────────────────
# Cek config file Co-DETR
# ─────────────────────────────────────────────
log_step "2. Deteksi Config Co-DETR"

# Cari config yang tersedia di repo yang di-clone
CODETR_CONFIG=""
POSSIBLE_CONFIGS=(
    "Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py"
    "Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_lsj_1x_coco.py"
    "Co-DETR/configs/co_detr/co_detr_r50_fpn_1x_coco.py"
)
for cfg in "${POSSIBLE_CONFIGS[@]}"; do
    if [ -f "$cfg" ]; then
        CODETR_CONFIG="$cfg"
        log_info "Config Co-DETR ditemukan: $cfg"
        break
    fi
done

if [ -z "$CODETR_CONFIG" ]; then
    log_error "Config Co-DETR tidak ditemukan! Pastikan repo sudah di-clone:"
    log_error "  git clone https://github.com/Sense-X/Co-DETR"
    exit 1
fi

# Update _base_ di config eksperimen
sed -i "s|../Co-DETR/projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py|../$CODETR_CONFIG|g" \
    configs/co_detr_r50_fp16_experiment.py \
    configs/co_detr_r50_fp32_baseline.py 2>/dev/null || true

# ─────────────────────────────────────────────
# Kondisi A — FP16
# ─────────────────────────────────────────────
log_step "3. Kondisi A: FP16 (Dynamic Loss Scaling)"

mkdir -p "$RESULTS_DIR/conditionA_fp16"
START_A=$(date +%s)

python train_with_monitor.py \
    --config "configs/co_detr_r50_fp16_experiment.py" \
    --condition fp16 \
    --max-iters "$MAX_ITERS" \
    --log-dir "$RESULTS_DIR/conditionA_fp16" \
    --data-root "$DATA_ROOT" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$RESULTS_DIR/conditionA_fp16/training_stdout.txt"

END_A=$(date +%s)
DURATION_A=$((END_A - START_A))
log_info "Kondisi A selesai dalam ${DURATION_A}s"

# Ringkasan cepat
if [ -f "$RESULTS_DIR/conditionA_fp16/overflow_log.csv" ]; then
    N_OVERFLOW=$(python -c "
import csv
with open('$RESULTS_DIR/conditionA_fp16/overflow_log.csv') as f:
    r = list(csv.DictReader(f))
print(sum(1 for row in r if row.get('is_inf','False')=='True' or row.get('is_nan','False')=='True'))
" 2>/dev/null || echo "?")
    log_info "Kondisi A — total overflow events: $N_OVERFLOW"
fi

# ─────────────────────────────────────────────
# Kondisi B — FP32
# ─────────────────────────────────────────────
log_step "4. Kondisi B: FP32 (Baseline)"

mkdir -p "$RESULTS_DIR/conditionB_fp32"
START_B=$(date +%s)

python train_with_monitor.py \
    --config "configs/co_detr_r50_fp32_baseline.py" \
    --condition fp32 \
    --max-iters "$MAX_ITERS" \
    --log-dir "$RESULTS_DIR/conditionB_fp32" \
    --data-root "$DATA_ROOT" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$RESULTS_DIR/conditionB_fp32/training_stdout.txt"

END_B=$(date +%s)
DURATION_B=$((END_B - START_B))
log_info "Kondisi B selesai dalam ${DURATION_B}s"

# ─────────────────────────────────────────────
# Analisis & Plot
# ─────────────────────────────────────────────
log_step "5. Analisis & Visualisasi"

mkdir -p "$RESULTS_DIR/plots"
python analysis/plot_results.py \
    --fp16-dir "$RESULTS_DIR/conditionA_fp16" \
    --fp32-dir "$RESULTS_DIR/conditionB_fp32" \
    --output-dir "$RESULTS_DIR/plots"

log_step "6. Generate Laporan"
python analysis/generate_report.py \
    --fp16-dir "$RESULTS_DIR/conditionA_fp16" \
    --fp32-dir "$RESULTS_DIR/conditionB_fp32" \
    --output "REPORT.md"

# ─────────────────────────────────────────────
# Ringkasan final
# ─────────────────────────────────────────────
log_step "✅ EKSPERIMEN SELESAI"
echo ""
log_info "Hasil tersimpan di:"
log_info "  └── $RESULTS_DIR/"
log_info "       ├── conditionA_fp16/  → overflow_log.csv, gradscaler_log.csv"
log_info "       ├── conditionB_fp32/  → overflow_log.csv"
log_info "       └── plots/            → *.png"
log_info "  └── REPORT.md"
echo ""
log_info "Transfer ke PC Windows dengan:"
log_info "  scp -r user@pc_training:/path/to/project/results/ ."
