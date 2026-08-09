#!/bin/bash
# =============================================================================
# install.sh — Setup environment untuk eksperimen Co-DETR FP16 Overflow
# Target: Pop!_OS/Ubuntu, NVIDIA RTX 5060 Ti (sm_120), CUDA 13.2
# =============================================================================
set -e  # exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ─────────────────────────────────────────────
# 0. Verifikasi NVIDIA driver & CUDA
# ─────────────────────────────────────────────
log_info "=== Tahap 0: Verifikasi NVIDIA Driver & CUDA ==="
if ! command -v nvidia-smi &>/dev/null; then
    log_error "nvidia-smi tidak ditemukan. Pastikan driver NVIDIA terinstall."
    exit 1
fi

DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
log_info "Driver NVIDIA: $DRIVER_VER"

# CUDA dari nvidia-smi
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
log_info "CUDA Version (driver): $CUDA_VER"

# Cek compute capability
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
log_info "Compute Capability GPU: sm_${CC/./}"

# ─────────────────────────────────────────────
# 1. Buat dan aktifkan Conda environment
# ─────────────────────────────────────────────
log_info "=== Tahap 1: Setup Python Environment ==="

ENV_NAME="codetr_fp16_exp"

if conda env list | grep -q "^$ENV_NAME "; then
    log_warn "Environment '$ENV_NAME' sudah ada. Skip pembuatan."
else
    log_info "Membuat conda environment '$ENV_NAME' dengan Python 3.10..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Aktifkan env (untuk script, kita source conda)
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
log_info "Environment aktif: $(conda info --envs | grep '*')"

# Pastikan setuptools<80 dan wheel terinstall (setuptools >=80 menghapus pkg_resources!)
pip install --upgrade pip "setuptools<80" wheel -q

# ─────────────────────────────────────────────
# 2. Install PyTorch (dengan dukungan sm_120)
# ─────────────────────────────────────────────
log_info "=== Tahap 2: Install PyTorch untuk sm_120 ==="

# Cek apakah PyTorch sudah ada dan mendukung sm_120
PYTORCH_OK=false
if python -c "import torch; cc=torch.cuda.get_device_capability(); exit(0 if cc>=(12,0) else 1)" 2>/dev/null; then
    log_info "PyTorch sudah ada dan mendukung sm_120."
    PYTORCH_OK=true
fi

if [ "$PYTORCH_OK" = false ]; then
    log_info "Menginstall PyTorch untuk CUDA 12.8 (mendukung sm_120)..."
    # Coba versi stable cu128 / cu126 terlebih dahulu
    if ! pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128; then
        log_warn "Stable cu128 gagal, mencoba nightly build tanpa torchaudio..."
        pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    fi
fi

# Verifikasi
log_info "Verifikasi PyTorch:"
python -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    print(f'  Compute Cap.    : {cc}')
    if cc[0] < 12:
        print('[WARN] sm_120 TIDAK terdukung. Hasil mungkin tidak representatif.')
    else:
        print('  [OK] sm_120 terdukung.')
"

# ─────────────────────────────────────────────
# 3. Clone Co-DETR
# ─────────────────────────────────────────────
log_info "=== Tahap 3: Clone repo Co-DETR ==="

REPO_DIR="./Co-DETR"
if [ -d "$REPO_DIR" ]; then
    log_warn "Direktori '$REPO_DIR' sudah ada. Skip clone."
else
    git clone https://github.com/Sense-X/Co-DETR "$REPO_DIR"
fi

cd "$REPO_DIR"
log_info "Masuk ke direktori: $(pwd)"

# ─────────────────────────────────────────────
# 4. Install mmcv (coba pre-built, fallback source)
# ─────────────────────────────────────────────
log_info "=== Tahap 4: Install mmcv ==="

TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
TORCH_MAJOR=$(echo $TORCH_VERSION | cut -d'.' -f1)
TORCH_MINOR=$(echo $TORCH_VERSION | cut -d'.' -f2)

log_info "PyTorch version (untuk mmcv): $TORCH_VERSION"

# Coba install mmcv dari openmim (paling mudah)
pip install openmim -q
log_info "Mencoba install mmcv via openmim..."

MMCV_INSTALLED=false
if mim install "mmcv>=2.0.0" 2>/dev/null; then
    log_info "mmcv berhasil diinstall via openmim."
    MMCV_INSTALLED=true
fi

# Fallback: build dari source jika pre-built tidak tersedia untuk kombinasi ini
if [ "$MMCV_INSTALLED" = false ]; then
    log_warn "Pre-built mmcv tidak tersedia. Build dari source..."
    log_warn "Proses ini memakan waktu 10-30 menit, harap bersabar."

    # Pastikan nvcc tersedia
    if ! command -v nvcc &>/dev/null; then
        log_error "nvcc tidak ditemukan! Install CUDA Toolkit:"
        log_error "  sudo apt install nvidia-cuda-toolkit"
        log_error "  atau download dari https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi

    log_info "nvcc version: $(nvcc --version | grep release)"

    # Clone dan build mmcv dari source
    cd ..
    if [ ! -d "mmcv_src" ]; then
        git clone https://github.com/open-mmlab/mmcv mmcv_src
    fi
    cd mmcv_src
    pip install "setuptools<80" wheel -q
    pip install -r requirements.txt -q
    MMCV_WITH_OPS=1 pip install --no-build-isolation -e . -v
    cd ../Co-DETR
    MMCV_INSTALLED=true
fi

# ─────────────────────────────────────────────
# 5. Install dependency Co-DETR
# ─────────────────────────────────────────────
log_info "=== Tahap 5: Install dependency Co-DETR ==="

pip install -r requirements.txt -q || log_warn "Beberapa dependency mungkin gagal — cek output di atas."

# mmdetection
if ! python -c "import mmdet" 2>/dev/null; then
    pip install mmdet -q
fi

# Dependency tambahan untuk eksperimen
pip install matplotlib pandas scipy tqdm pycocotools -q

# ─────────────────────────────────────────────
# 6. Verifikasi final
# ─────────────────────────────────────────────
log_info "=== Tahap 6: Verifikasi Final ==="
cd ..
python setup/verify_env.py

log_info "============================================"
log_info "SETUP SELESAI!"
log_info "Langkah berikutnya:"
log_info "  1. conda activate $ENV_NAME"
log_info "  2. python data/prepare_dataset.py"
log_info "  3. bash run_experiment.sh"
log_info "============================================"
