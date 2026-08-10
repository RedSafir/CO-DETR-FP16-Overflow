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
python -c "import torch; assert torch.cuda.is_available(), 'CUDA tidak tersedia!'" && \
    log_info "PyTorch CUDA tersedia" || { log_error "PyTorch CUDA tidak tersedia!"; exit 1; }

# ─────────────────────────────────────────────
# Script standalone — tidak butuh dataset/config
# ─────────────────────────────────────────────
log_step "1. Mode Standalone (Pure PyTorch, no mmcv/mmdet)"
log_info "Menggunakan train_standalone.py — arsitektur Co-DETR replika."
log_info "Tidak perlu dataset COCO atau mmcv. Berjalan dengan data sintetis."

# ─────────────────────────────────────────────
# Verifikasi PyTorch + CUDA
# ─────────────────────────────────────────────
log_step "2. Verifikasi PyTorch & CUDA"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')
print(f'  FP16 ok  : {torch.cuda.is_available()}')
"

# ─────────────────────────────────────────────
# Kondisi A — FP16
# ─────────────────────────────────────────────
log_step "3. Kondisi A: FP16 (Dynamic Loss Scaling)"

mkdir -p "$RESULTS_DIR/conditionA_fp16"
START_A=$(date +%s)

python train_standalone.py \
    --condition fp16 \
    --max-iters "$MAX_ITERS" \
    --log-dir "$RESULTS_DIR/conditionA_fp16" \
    --batch-size "$BATCH_SIZE" \
    --seed 42 \
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
print(sum(1 for row in r if str(row.get('is_inf','False'))=='True' or str(row.get('is_nan','False'))=='True'))
" 2>/dev/null || echo "?")
    log_info "Kondisi A — total overflow events: $N_OVERFLOW"
fi

# ─────────────────────────────────────────────
# Kondisi B — FP32
# ─────────────────────────────────────────────
log_step "4. Kondisi B: FP32 (Baseline)"

mkdir -p "$RESULTS_DIR/conditionB_fp32"
START_B=$(date +%s)

python train_standalone.py \
    --condition fp32 \
    --max-iters "$MAX_ITERS" \
    --log-dir "$RESULTS_DIR/conditionB_fp32" \
    --batch-size "$BATCH_SIZE" \
    --seed 42 \
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
