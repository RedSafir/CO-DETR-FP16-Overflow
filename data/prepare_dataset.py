"""
prepare_dataset.py — Download dan siapkan subset COCO val2017 (500 gambar)
untuk eksperimen Co-DETR FP16 Overflow.

Output:
  data/coco_subset/
  ├── images/
  │   └── val2017/      ← 500 gambar JPEG
  └── annotations/
      └── instances_val2017_subset.json  ← annotation COCO format (subset)
"""
import os
import json
import shutil
import random
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    import urllib.request as urlreq
except ImportError:
    import urllib as urlreq

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
COCO_VAL_IMG_URL  = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL      = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
OUTPUT_DIR        = Path("data/coco_subset")
N_IMAGES          = 500
RANDOM_SEED       = 42

# Pilih gambar yang banyak objek kecil → lebih cepat memicu overflow logit besar
# COCO val2017 punya banyak gambar semacam itu — kita pilih berdasarkan jumlah ann terbanyak
MIN_ANNOTATIONS   = 5  # filter gambar dengan setidaknya 5 objek


def download_with_progress(url: str, dest: Path):
    """Download file dengan progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} sudah ada.")
        return

    print(f"  Downloading {dest.name} dari {url} ...")
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)

    urlreq.urlretrieve(url, str(dest), reporthook=hook)
    print()  # newline setelah progress


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract ZIP file."""
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  Extracted ke {dest_dir}")


def create_subset_annotations(
    full_ann_path: Path,
    image_dir: Path,
    output_dir: Path,
    n_images: int,
    min_ann: int,
    seed: int
) -> dict:
    """
    Buat subset annotation COCO dari full val2017.
    Prioritaskan gambar dengan banyak annotations (lebih menarik untuk overflow).
    """
    print(f"\n  Membaca annotation penuh dari {full_ann_path}...")
    with open(full_ann_path) as f:
        full = json.load(f)

    # Hitung jumlah annotation per gambar
    ann_count = {}
    for ann in full["annotations"]:
        img_id = ann["image_id"]
        ann_count[img_id] = ann_count.get(img_id, 0) + 1

    # Filter gambar yang tersedia di disk dan punya cukup annotation
    available_images = []
    for img_info in full["images"]:
        img_path = image_dir / img_info["file_name"]
        n_ann = ann_count.get(img_info["id"], 0)
        if img_path.exists() and n_ann >= min_ann:
            available_images.append((img_info, n_ann))

    print(f"  Gambar tersedia dengan >= {min_ann} objek: {len(available_images)}")

    # Sort by annotation count descending, then shuffle untuk variety
    available_images.sort(key=lambda x: -x[1])
    top_half = available_images[:len(available_images)//2]
    random.seed(seed)
    random.shuffle(top_half)
    selected = top_half[:n_images]

    selected_img_infos = [x[0] for x in selected]
    selected_ids = {x[0]["id"] for x in selected}

    # Filter annotations yang relevan
    selected_anns = [a for a in full["annotations"] if a["image_id"] in selected_ids]

    subset = {
        "info":        full.get("info", {}),
        "licenses":    full.get("licenses", []),
        "categories":  full["categories"],
        "images":      selected_img_infos,
        "annotations": selected_anns,
    }

    # Simpan annotation subset
    output_dir.mkdir(parents=True, exist_ok=True)
    out_ann = output_dir / "annotations" / "instances_val2017_subset.json"
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ann, "w") as f:
        json.dump(subset, f)

    total_ann = len(selected_anns)
    avg_ann = total_ann / len(selected_img_infos) if selected_img_infos else 0
    print(f"  Subset: {len(selected_img_infos)} gambar, {total_ann} annotations "
          f"(rata-rata {avg_ann:.1f} per gambar)")
    print(f"  Disimpan ke: {out_ann}")

    return subset


def copy_subset_images(selected_images: list, src_dir: Path, dst_dir: Path):
    """Copy gambar subset ke direktori tujuan."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Menyalin {len(selected_images)} gambar ke {dst_dir}...")
    for img_info in tqdm(selected_images, desc="  Copy images"):
        src = src_dir / img_info["file_name"]
        dst = dst_dir / img_info["file_name"]
        if not dst.exists():
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO val2017 subset")
    parser.add_argument("--n-images", type=int, default=N_IMAGES,
                        help=f"Jumlah gambar subset (default: {N_IMAGES})")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Direktori output")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download jika file sudah ada")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  PERSIAPAN DATASET COCO val2017 SUBSET")
    print("="*60)

    # ── 1. Download annotation ──────────────────────────────────────────────
    print("\n[1/4] Download annotation COCO...")
    ann_zip = tmp_dir / "annotations_trainval2017.zip"
    if not args.skip_download:
        download_with_progress(COCO_ANN_URL, ann_zip)

    ann_extract_dir = tmp_dir / "annotations_extracted"
    if not (ann_extract_dir / "annotations").exists():
        extract_zip(ann_zip, ann_extract_dir)

    full_ann_path = ann_extract_dir / "annotations" / "instances_val2017.json"
    if not full_ann_path.exists():
        raise FileNotFoundError(f"Annotation tidak ditemukan: {full_ann_path}")

    # ── 2. Buat subset annotation (untuk tahu gambar mana yang dipilih) ──────
    print("\n[2/4] Membuat subset annotation...")
    # Load dulu untuk tahu gambar mana yang dipilih
    with open(full_ann_path) as f:
        full_ann = json.load(f)

    ann_count = {}
    for ann in full_ann["annotations"]:
        ann_count[ann["image_id"]] = ann_count.get(ann["image_id"], 0) + 1

    # Daftar semua image_id (kita belum tahu yang mana ada di disk)
    # Download gambar berdasarkan subset yang dipilih
    random.seed(RANDOM_SEED)
    candidates = [(img, ann_count.get(img["id"], 0))
                  for img in full_ann["images"]
                  if ann_count.get(img["id"], 0) >= MIN_ANNOTATIONS]
    candidates.sort(key=lambda x: -x[1])
    top_half = candidates[:len(candidates)//2]
    random.shuffle(top_half)
    selected = top_half[:args.n_images]
    selected_img_infos = [x[0] for x in selected]
    selected_ids = {x[0]["id"] for x in selected}

    # ── 3. Download HANYA gambar yang dipilih (lebih efisien dari download 6GB) ──
    print(f"\n[3/4] Download {len(selected_img_infos)} gambar COCO val2017...")
    img_dir = output_dir / "images" / "val2017"
    img_dir.mkdir(parents=True, exist_ok=True)

    COCO_IMG_BASE = "http://images.cocodataset.org/val2017"
    failed = []
    for img_info in tqdm(selected_img_infos, desc="  Downloading images"):
        fname = img_info["file_name"]
        dst = img_dir / fname
        if dst.exists():
            continue
        url = f"{COCO_IMG_BASE}/{fname}"
        try:
            urlreq.urlretrieve(url, str(dst))
        except Exception as e:
            failed.append((fname, str(e)))

    if failed:
        print(f"  [WARN] {len(failed)} gambar gagal didownload:")
        for fname, err in failed[:5]:
            print(f"    {fname}: {err}")

    # ── 4. Simpan annotation subset ─────────────────────────────────────────
    print("\n[4/4] Menyimpan annotation subset...")

    # Filter hanya gambar yang berhasil didownload
    available_imgs = [img for img in selected_img_infos
                      if (img_dir / img["file_name"]).exists()]
    available_ids = {img["id"] for img in available_imgs}
    selected_anns = [a for a in full_ann["annotations"]
                     if a["image_id"] in available_ids]

    subset = {
        "info":        full_ann.get("info", {}),
        "licenses":    full_ann.get("licenses", []),
        "categories":  full_ann["categories"],
        "images":      available_imgs,
        "annotations": selected_anns,
    }

    ann_out_dir = output_dir / "annotations"
    ann_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = ann_out_dir / "instances_val2017_subset.json"
    with open(out_path, "w") as f:
        json.dump(subset, f)

    avg_ann = len(selected_anns) / len(available_imgs) if available_imgs else 0
    print(f"\n{'='*60}")
    print(f"  DATASET SIAP!")
    print(f"  Gambar : {len(available_imgs)}")
    print(f"  Annotations: {len(selected_anns)} (avg {avg_ann:.1f}/gambar)")
    print(f"  Direktori  : {output_dir.resolve()}")
    print(f"{'='*60}\n")

    # Cetak config path untuk digunakan di training config
    print("  Gunakan path berikut di config mmdetection:")
    print(f"    data_root = '{output_dir.resolve()}'")
    print(f"    ann_file  = 'annotations/instances_val2017_subset.json'")
    print(f"    img_prefix = 'images/val2017/'")


if __name__ == "__main__":
    main()
