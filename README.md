# Co-DETR FP16 Overflow Experiment

Paket eksperimen untuk membuktikan secara empiris bahwa Co-DETR mengalami overflow numerik saat dilatih dengan FP16.

## Struktur Direktori

```
co_detr/
├── setup/
│   ├── install.sh          ← Script setup environment (jalankan PERTAMA)
│   └── verify_env.py       ← Verifikasi environment sebelum eksperimen
│
├── data/
│   └── prepare_dataset.py  ← Download COCO val2017 subset 500 gambar
│
├── instrumentation/
│   └── overflow_monitor.py ← Modul inti monitoring (hook + CSV logging)
│
├── configs/
│   ├── co_detr_r50_fp16_experiment.py  ← Kondisi A: FP16
│   └── co_detr_r50_fp32_baseline.py   ← Kondisi B: FP32
│
├── analysis/
│   ├── plot_results.py      ← Generate 4 plot perbandingan
│   └── generate_report.py  ← Generate REPORT.md dari CSV log
│
├── train_with_monitor.py   ← Script training utama (menggantikan tools/train.py)
├── run_experiment.sh       ← Master runner (jalankan semua kondisi sekaligus)
└── README.md               ← File ini
```

## Cara Pakai

### 1. Transfer ke PC Training

Salin seluruh folder ini ke PC training (Linux/Pop!_OS), lalu:

```bash
cd /path/to/co_detr
```

### 2. Setup Environment

```bash
bash setup/install.sh
```

Script ini akan:
- Membuat conda environment `codetr_fp16_exp`
- Install PyTorch dengan dukungan sm_120 (RTX 5060 Ti)
- Clone repo Co-DETR
- Install mmcv (pre-built atau build dari source)
- Install semua dependency

### 3. Siapkan Dataset

```bash
conda activate codetr_fp16_exp
python data/prepare_dataset.py --n-images 500 --output-dir data/coco_subset
```

### 4. Jalankan Eksperimen (Otomatis)

```bash
bash run_experiment.sh
```

Ini akan menjalankan:
- **Kondisi A** — FP16 training (300 iter)
- **Kondisi B** — FP32 baseline (300 iter)
- **Analisis** — generate plot dan laporan

### 4b. Jalankan Manual (Per Kondisi)

```bash
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
```

### 5. Analisis Hasil

```bash
# Generate plot
python analysis/plot_results.py \
  --fp16-dir results/conditionA_fp16 \
  --fp32-dir results/conditionB_fp32 \
  --output-dir results/plots

# Generate laporan
python analysis/generate_report.py \
  --fp16-dir results/conditionA_fp16 \
  --fp32-dir results/conditionB_fp32 \
  --output REPORT.md
```

### 6. Transfer Hasil ke Windows

```bash
# Dari Windows PowerShell:
scp -r user@pc_training:/path/to/co_detr/results/ .
scp user@pc_training:/path/to/co_detr/REPORT.md .
```

## Output yang Dihasilkan

```
results/
├── conditionA_fp16/
│   ├── overflow_log.csv       ← Log setiap tensor: step, max_abs, is_inf, is_nan
│   ├── gradscaler_log.csv     ← Log GradScaler: kapan scale turun (overflow gradien)
│   ├── training_log.csv       ← Loss per step
│   ├── summary.txt            ← Ringkasan teks
│   └── training_stdout.txt    ← Output lengkap training
│
├── conditionB_fp32/
│   ├── overflow_log.csv
│   ├── training_log.csv
│   └── training_stdout.txt
│
└── plots/
    ├── fig1_max_abs_value_vs_step.png   ← Kunci: A vs B, dengan garis FP16_MAX=65504
    ├── fig2_loss_scale_vs_step.png      ← GradScaler scale history
    ├── fig3_component_breakdown.png     ← Bar chart overflow per komponen
    └── fig4_dashboard_summary.png      ← Dashboard 4-panel ringkasan

REPORT.md  ← Laporan akhir lengkap (Markdown)
```

## Cara Membaca Hasil

### overflow_log.csv

| Kolom | Deskripsi |
|-------|-----------|
| `step` | Iterasi training |
| `component` | Layer yang dimonitor (misal: `AuxHead_ATSS/model.rpn_head`) |
| `max_abs_value` | Nilai absolut maksimum tensor saat itu |
| `is_inf` | `True` jika ada nilai Inf (overflow FP16 pasti) |
| `is_nan` | `True` jika ada nilai NaN (propagasi dari Inf) |
| `loss_scale` | Nilai GradScaler saat itu |
| `pct_of_fp16_max` | Persentase dari batas FP16 (65504) |

### gradscaler_log.csv

| Kolom | Deskripsi |
|-------|-----------|
| `overflow_detected` | `True` jika gradien overflow terdeteksi → step di-skip |
| `scale_before` | Scale sebelum step |
| `scale_after` | Scale setelah step (turun jika overflow) |

## Catatan Kompatibilitas sm_120

- **PyTorch**: Harus versi nightly dengan indeks cu128. Script install.sh menangani ini otomatis.
- **mmcv**: Jika wheel tidak tersedia, script akan build dari source (~20-30 menit).
- **CUDA**: Driver harus mendukung CUDA 12.8+. CUDA 13.2 yang terpasang sudah kompatibel.

## Pemecahan Masalah

### Error: `CUDA kernel image not available for sm_120`
→ PyTorch yang terinstall tidak mendukung sm_120. Jalankan lagi `setup/install.sh`.

### Error: `mmcv ops not found` atau `No module named mmcv._ext`
→ mmcv perlu di-build dari source. Script install.sh akan mencoba ini otomatis.

### Error: `Config file not found`
→ Repo Co-DETR belum di-clone. Jalankan `setup/install.sh` dari awal.

### OOM (Out of Memory)
→ Kurangi `--batch-size` ke 1 di perintah `train_with_monitor.py`.
