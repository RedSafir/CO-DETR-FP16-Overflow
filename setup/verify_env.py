"""
verify_env.py — Verifikasi environment sebelum eksperimen Co-DETR FP16 Overflow
"""
import sys

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []

def check(name, fn):
    try:
        msg = fn()
        results.append((PASS, name, msg))
        return True
    except Exception as e:
        results.append((FAIL, name, str(e)))
        return False

# ── 1. Python version ──────────────────────────────────────────────────────────
def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        raise RuntimeError(f"Python {v.major}.{v.minor} terlalu lama, butuh >= 3.9")
    return f"Python {v.major}.{v.minor}.{v.micro}"

check("Python >= 3.9", check_python)

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
def check_torch():
    import torch
    return f"torch {torch.__version__}"

check("PyTorch importable", check_torch)

# ── 3. CUDA available ─────────────────────────────────────────────────────────
def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() = False")
    return f"CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}"

check("CUDA available", check_cuda)

# ── 4. Compute capability ─────────────────────────────────────────────────────
def check_cc():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA")
    major, minor = torch.cuda.get_device_capability()
    cc = f"sm_{major}{minor}"
    if major < 7:
        raise RuntimeError(f"Compute cap {cc} terlalu lama — tidak mendukung Tensor Core FP16")
    if major < 12:
        return f"{cc} (FP16 Tensor Core OK, tapi bukan Blackwell — WARNING: sm_120 check gagal)"
    return f"{cc} (Blackwell — OK)"

check("Compute Capability >= sm_70", check_cc)

# ── 5. FP16 tensor op ─────────────────────────────────────────────────────────
def check_fp16():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA")
    a = torch.randn(64, 64, device='cuda', dtype=torch.float16)
    b = torch.randn(64, 64, device='cuda', dtype=torch.float16)
    c = torch.matmul(a, b)
    import math
    if math.isnan(c.max().item()):
        raise RuntimeError("FP16 matmul menghasilkan NaN")
    return f"FP16 matmul OK, shape {c.shape}"

check("FP16 matmul functional", check_fp16)

# ── 6. AMP GradScaler ─────────────────────────────────────────────────────────
def check_amp():
    import torch
    scaler = torch.cuda.amp.GradScaler()
    return f"GradScaler initial scale: {scaler.get_scale()}"

check("torch.cuda.amp.GradScaler", check_amp)

# ── 7. mmcv ───────────────────────────────────────────────────────────────────
def check_mmcv():
    import mmcv
    return f"mmcv {mmcv.__version__}"

check("mmcv importable", check_mmcv)

# ── 8. mmdet ──────────────────────────────────────────────────────────────────
def check_mmdet():
    import mmdet
    return f"mmdet {mmdet.__version__}"

check("mmdet importable", check_mmdet)

# ── 9. Dependencies eksperimen ────────────────────────────────────────────────
def check_deps():
    import matplotlib, pandas, scipy, tqdm
    return f"matplotlib={matplotlib.__version__}, pandas={pandas.__version__}"

check("matplotlib/pandas/scipy/tqdm", check_deps)

# ── 10. Disk space (butuh ~5GB untuk COCO subset + model) ────────────────────
def check_disk():
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    if free_gb < 10:
        raise RuntimeError(f"Disk tersisa {free_gb:.1f}GB — butuh minimal 10GB")
    return f"Disk free: {free_gb:.1f}GB"

check("Disk space >= 10GB", check_disk)

# ── Print hasil ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  HASIL VERIFIKASI ENVIRONMENT")
print("="*60)
for status, name, msg in results:
    print(f"  {status} {name:<40} {msg}")

fail_count = sum(1 for s, _, _ in results if "FAIL" in s)
print("="*60)
if fail_count == 0:
    print("\033[92m  Semua cek PASSED — environment siap untuk eksperimen!\033[0m")
else:
    print(f"\033[91m  {fail_count} cek FAILED — perbaiki error di atas sebelum lanjut.\033[0m")
print("="*60 + "\n")

sys.exit(fail_count)
