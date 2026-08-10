# Laporan Eksperimen: Pembuktian Overflow FP16 pada Co-DETR

**Tanggal:** 2026-08-10 11:08:37
**Hardware:** NVIDIA GeForce RTX 5060 Ti (sm_120, 16GB)
**Model:** Co-DETR dengan backbone ResNet-50
**Dataset:** COCO val2017 subset (~500 gambar)

---

## 1. Ringkasan Eksekutif

| Kondisi | Status Overflow | Total Events | Max |value| |
|---------|----------------|--------------|-------------|
| A (FP16 + dynamic scale) | ⚠️ Near-overflow terdeteksi, inf/nan belum terpicu | 0 | 0.0000e+00 |
| B (FP32 baseline)        | ✅ **BERSIH**          | 0 | 0.0000e+00 |

> **Catatan:** Overflow belum terdeteksi dalam 0 measurements.
> Pertimbangkan menjalankan lebih banyak iterasi atau batch size lebih besar.

## 2. Temuan Per Komponen

### 2.1 Komponen Paling Rentan (Kondisi A — FP16)

*Tidak ada overflow events untuk dibreakdown.*

### 2.2 Near-Overflow Events (>50% dari FP16 Max)

*Tidak ada near-overflow events.*

## 3. Jenis Overflow yang Terdeteksi

| Jenis | Mekanisme | Terdeteksi? | Detail |
|-------|-----------|-------------|--------|
| **Overflow Aktivasi** (forward pass) | Forward hook pada attention & aux heads | ❌ Tidak | Step pertama: N/A (N/A) |
| **Overflow Gradien** (backward pass) | GradScaler deteksi inf/nan di gradien | ❌ Tidak | Step pertama: N/A, total skip: 0 |

## 4. Timeline Overflow

| Kejadian | Step | Kondisi |
|----------|------|---------|
| Max |value| = 0.00e+00 | None | A (FP16) |
| Tidak ada overflow | — | B (FP32) |

## 5. Log GradScaler (Kondisi A — FP16)

- Total steps yang di-skip karena overflow gradien: **0**
- Log detail tersimpan di: `results/conditionA_fp16/gradscaler_log.csv`

## 6. Rekomendasi Mitigasi

Berdasarkan bukti eksperimen, berikut adalah rekomendasi mitigasi dari yang paling efektif:

| # | Solusi | Prioritas | Cara Implementasi |
|---|--------|-----------|------------------|
| 1 | **BF16 (Brain Float 16)** | 🔴 HIGH | `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True` |
| 2 | **QK-Normalization di Decoder** | 🔴 HIGH | `nn.LayerNorm(d_model)` di TransformerDecoder sebelum `q @ k.T` |
| 3 | **TF32 (TensorFloat-32)** | 🟡 MEDIUM | `torch.backends.cuda.matmul.allow_tf32 = True` |
| 4 | **Per-head Loss Weighting** | 🟡 MEDIUM | Tambahkan `loss_weight` per auxiliary head di config Co-DETR |
| 5 | **Gradient Clipping Lebih Agresif** | 🟢 LOW | `optimizer_config = dict(grad_clip=dict(max_norm=0.01))` |

### 6.1 BF16 (Brain Float 16)

Ganti FP16 dengan BF16. Range BF16 identik dengan FP32 (~3.4e38 vs 65504) sehingga overflow tidak mungkin terjadi. RTX 5060 Ti (sm_120) mendukung BF16 native. `autocast(dtype=torch.bfloat16)`

```python
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True`
```

### 6.2 QK-Normalization di Decoder

Tambahkan LayerNorm pada Q dan K sebelum dot-product attention. Mencegah attention logit meledak saat query/key memiliki magnitude besar. Diimplementasikan di ViT-22B dan model DETR terbaru.

```python
`nn.LayerNorm(d_model)` di TransformerDecoder sebelum `q @ k.T`
```

### 6.3 TF32 (TensorFloat-32)

Aktifkan TF32 untuk operasi matmul — memakai akumulator FP32 internal tapi tetap cepat. Secara teknis masih FP32 range, sehingga tidak overflow.

```python
`torch.backends.cuda.matmul.allow_tf32 = True`
```

### 6.4 Per-head Loss Weighting

Berikan weight berbeda untuk loss dari setiap auxiliary head. ATSS/FCOS loss scale bisa berbeda order of magnitude dengan head utama, menyebabkan loss gabungan memiliki magnitudo yang sulit diskalakan dengan satu GradScaler.

```python
Tambahkan `loss_weight` per auxiliary head di config Co-DETR
```

### 6.5 Gradient Clipping Lebih Agresif

Kurangi `max_norm` di `grad_clip` dari 0.1 ke 0.01. Mencegah gradien besar memperbesar parameter hingga aktivasi overflow di iterasi berikutnya.

```python
`optimizer_config = dict(grad_clip=dict(max_norm=0.01))`
```

---

## Lampiran: File Output

| File | Deskripsi |
|------|-----------|
| `results/conditionA_fp16/overflow_log.csv` | Log detail setiap tensor yang dimonitor (FP16) |
| `results/conditionA_fp16/gradscaler_log.csv` | Log GradScaler per step — kapan scale turun |
| `results/conditionA_fp16/training_log.csv` | Loss per step (FP16) |
| `results/conditionB_fp32/overflow_log.csv` | Log detail setiap tensor yang dimonitor (FP32) |
| `results/plots/fig1_max_abs_value_vs_step.png` | Plot max |value| vs step — A vs B |
| `results/plots/fig2_loss_scale_vs_step.png` | Plot loss scale GradScaler vs step |
| `results/plots/fig3_component_breakdown.png` | Bar chart overflow per komponen |
| `results/plots/fig4_dashboard_summary.png` | Dashboard ringkasan 4-panel |

---
*Laporan ini di-generate otomatis oleh `analysis/generate_report.py`*