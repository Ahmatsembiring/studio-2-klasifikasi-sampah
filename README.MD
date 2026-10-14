# Studio 2 — Klasifikasi Sampah · Kelompok <KELOMPOK 1>

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap           | NIM       | Kontribusi utama |
|----|------------------------|-----------|------------------|
| 1  | Ahmat Prayoga Sembiring| 123140053 |                  |
| 2  | Tengku Hafid Diraputra  | 123140043 |                  |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny pretrained ImageNet-22k → fine-tuned ImageNet-1k (`timm`: `convnext_tiny.fb_in22k_ft_in1k`), head linear 5 kelas. Seluruh layer di-*fine-tune* (tanpa freezing).
- **Strategi pelatihan**:
  - Optimizer AdamW, LR backbone 1e-4, LR head 10× (1e-3), weight decay 0,05
  - Warmup linear 1 epoch, lalu cosine decay per-step; 15 epoch; batch 32; mixed precision (AMP)
  - Loss cross-entropy dengan *label smoothing* 0,1 (membantu menghadapi label noise)
  - Checkpoint dipilih berdasarkan **Macro-F1** pada split validasi lokal (metrik penilaian), dihitung dengan fungsi `metrics()` dari `evaluate.py` resmi
  - Augmentasi: `RandomResizedCrop(224, scale=0.4–1.0)`, horizontal flip, `TrivialAugmentWide`, `RandomErasing(p=0.25)`
- **Data tambahan**: tidak ada. Hanya train set resmi.
- **Ukuran input & preprocessing**: resize sisi pendek 256 → center crop 224 → normalisasi mean/std ImageNet. Konfigurasi disimpan di `submission/config.json` dan dibaca oleh `get_transform()`, sehingga preprocessing evaluasi selalu sama dengan saat training.
- **Bobot**: disimpan dalam fp16 (55,7 MB) agar di bawah batas 100 MB GitHub. Selisih prediksi dibanding fp32 diuji < 0,0002 probabilitas, dengan argmax identik.

## Hasil

Split validasi lokal: 15% per kelas, seed 42 (fungsi `make_local_split`, identik dengan `train_baseline.py`).

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `<hash>` | 0,9722 | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

| ID | Konfigurasi | Macro-F1 lokal | Catatan |
|----|-------------|----------------|---------|
| E0 | Baseline ResNet-18 (starter kit, 8 epoch) | — | referensi: ~0,92 di validation set tersembunyi (halaman kompetisi) |
| E1 | ConvNeXt-Tiny in22k, 15 epoch, aug strong | **0,9722** (epoch 12) | F1 terendah `kertas_kardus` 0,9615. Kesalahan dominan: keluarga gelas (plastik / kertas / styrofoam), yang bentuknya sama dan hanya berbeda material. Epoch 9–15 setara (0,963–0,972, dalam batas noise ±1 gambar). |

- E1 dilatih dengan 2.753 dari 2.964 gambar karena salinan dataset di Drive belum lengkap. Akan diulang dengan data lengkap.
- Audit label dari `train_sources.csv` (`tools/audit_sources.py`) menemukan 40 kandidat label keliru (`audit/suspects.csv`). Salah satunya ("Polystyrene cups" berlabel `styrofoam`) muncul juga sebagai kesalahan model di validasi.

## Cara Reproduksi

Dataset tidak disertakan di repo (sesuai aturan kompetisi). Letakkan folder `train/` dari Google Drive asisten di luar repo.

**Google Colab (yang dipakai kelompok):** buka `notebooks/studio2_kasus_final.ipynb`, pilih runtime T4 GPU, ubah `DATA_SRC`, lalu *Run all*. Notebook menjalankan Bagian A (validasi), B (data penuh), dan C (paket submission).

**Lokal:**

```bash
pip install -r requirements.txt
python train.py --data ../data/train --backbone convnext_tiny.fb_in22k_ft_in1k --epochs 15 --name A_cnxt_t
# salin model.pth + config.json dari runs/A_cnxt_t/submission/ ke submission/
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform() — membaca config.json
  config.json       # backbone + ukuran input + mean/std
  model.pth         # state_dict model terbaik (fp16, 55,7 MB)
  requirements.txt  # timm==1.0.30
evaluate.py         # salinan skrip evaluasi resmi
train.py            # training: seleksi Macro-F1, log, ekspor submission
train_baseline.py   # baseline dari starter kit
notebooks/          # notebook Colab (kasus -> final -> paket)
tools/              # audit label dari train_sources.csv
audit/              # hasil audit (kandidat label keliru)
```
