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

# Set pip timeout ke 1000 detik agar tidak putus di tengah jalan saat download file besar (seperti CUDNN ~650MB)
export PIP_DEFAULT_TIMEOUT=1000

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_ROOT="$(pwd)"

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

# Pastikan setuptools==69.5.1 terinstall (menyediakan pkg_resources dan kompatibel dengan openxlab)
pip install --upgrade pip "setuptools==69.5.1" wheel -q

# ─────────────────────────────────────────────
# 2. Install PyTorch (dengan dukungan sm_120)
# ─────────────────────────────────────────────
log_info "=== Tahap 2: Install PyTorch untuk sm_120 ==="

PYTORCH_OK=false
if python -c "import torch; cc=torch.cuda.get_device_capability(); exit(0 if cc>=(12,0) else 1)" 2>/dev/null; then
    log_info "PyTorch sudah ada dan mendukung sm_120."
    PYTORCH_OK=true
fi

if [ "$PYTORCH_OK" = false ]; then
    log_info "Menginstall PyTorch Stable cu124 (kompatibel dengan CUDA 12.8)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
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
    pip install "setuptools==69.5.1" wheel -q

    # Install build dependency mmcv (file requirements bisa berbeda per versi)
    pip install ninja -q
    [ -f "requirements/optional.txt" ] && pip install -r requirements/optional.txt -q
    [ -f "requirements.txt" ] && pip install -r requirements.txt -q

    # ── Strategi bypass g++ 13.3.0 vs CUDA 12.0 ──────────────────────────────
    # Masalah: dua cek yang harus dibypass SEKALIGUS:
    #   1. PyTorch Python check: baca output `g++ --version` → tolak jika > 12.x
    #   2. CUDA 12.0 C header check: #if __GNUC__ > 12 → compile-time error
    #
    # Solusi: fake g++ wrapper (laporan versi 12.9) + -allow-unsupported-compiler
    # ─────────────────────────────────────────────────────────────────────────────

    GPP_WRAPPER_DIR="$HOME/.local/bin"
    mkdir -p "$GPP_WRAPPER_DIR"

    cat > "$GPP_WRAPPER_DIR/g++" << 'WRAPPER_EOF'
#!/bin/bash
# Fake g++ wrapper: lapor versi 12.9 ke PyTorch/CUDA check,
# tapi pakai g++ 13.3.0 asli untuk kompilasi nyata
if [[ "$1" == "--version" ]] || [[ "$*" == "--version" ]]; then
    echo "g++ (Ubuntu 12.9.0-compat-wrapper) 12.9.0"
    echo "Copyright (C) 2022 Free Software Foundation, Inc."
    exit 0
fi
exec /usr/bin/g++ "$@"
WRAPPER_EOF

    chmod +x "$GPP_WRAPPER_DIR/g++"
    log_info "Fake g++ wrapper dibuat di $GPP_WRAPPER_DIR/g++ (lapor versi 12.9 ke CUDA)"

    # Tambahkan wrapper ke PATH (harus di depan /usr/bin)
    export PATH="$GPP_WRAPPER_DIR:$PATH"

    # Bypass CUDA 12.0 C header check (#if __GNUC__ > 12)
    export NVCC_PREPEND_FLAGS="-D__CUDA_ALLOW_UNSUPPORTED_COMPILER__ -allow-unsupported-compiler"
    export CXXFLAGS="-D__CUDA_ALLOW_UNSUPPORTED_COMPILER__"

    log_info "PATH=$PATH"
    log_info "NVCC_PREPEND_FLAGS=$NVCC_PREPEND_FLAGS"
    log_info "Verifikasi g++ yang aktif: $(which g++) → $(g++ --version | head -1)"

    MMCV_WITH_OPS=1 pip install --no-build-isolation . -v
    cd "$PROJECT_ROOT"
    MMCV_INSTALLED=true
fi

# ─────────────────────────────────────────────
# 5. Install dependency Co-DETR
# ─────────────────────────────────────────────
log_info "=== Tahap 5: Install Co-DETR Package & Dependency ==="

if [ -d "Co-DETR" ]; then
    # Patch mmcv_maximum_version agar mengizinkan mmcv 2.x
    sed -i "s/mmcv_maximum_version = '1.7.0'/mmcv_maximum_version = '2.3.0'/g" Co-DETR/mmdet/__init__.py 2>/dev/null || true
    cd Co-DETR
    python setup.py develop -q || pip install --no-build-isolation -e . -q || log_warn "Install Co-DETR package warning"
    cd "$PROJECT_ROOT"
else
    log_warn "Folder Co-DETR tidak ditemukan, menginstall mmdet standar..."
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
