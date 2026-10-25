# Studio 2 — Klasifikasi Sampah · KELOMPOK 1

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Ahmat Prayoga Sembiring | 123140053 | Setup repo & alur Git; menjalankan training/eksperimen di Google Colab (resolusi 224 vs 288, TTA); verifikasi kelengkapan data; pengemasan & pengujian submission; penulisan README |
| 2  | Tengku Hafid Diraputra | 123140043 | Perancangan pipeline pelatihan & pemilihan hiperparameter (`train.py`); implementasi modul inferensi, TTA flip, & ekspor `fp16` (`submission/model.py`); evaluasi metrik F1 per kelas, *confusion matrix*, & analisis kesalahan (*error analysis*); reproduksi & verifikasi akhir submission |

## Model Final

Model yang dikirim untuk Final adalah **model Preview 2**: ConvNeXt-Tiny 224 px dengan TTA flip horizontal, dilatih pada seluruh data yang tersedia (2.753 gambar). Isi `submission/` berasal dari commit `4daca6c`.

Model ini dipilih karena skor Preview 2 (**0,9675**) lebih tinggi dari Preview 1 (0,9595) pada validation set tersembunyi. Setelah Preview 2, kelompok **tidak mengirim model baru**. Model yang belum pernah dievaluasi pada validation set tersembunyi tidak dipakai untuk Final, karena skor validasi lokal terbukti optimis (lihat bagian Pelajaran).

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny pretrained ImageNet-22k → fine-tuned ImageNet-1k (`timm`: `convnext_tiny.fb_in22k_ft_in1k`), head linear 5 kelas. Seluruh layer di-*fine-tune* (tanpa freezing).
- **Strategi pelatihan**:
  - AdamW, LR backbone 0.0001, LR head 10×, weight decay 0,05; warmup linear 1 epoch lalu cosine decay per-step; 15 epoch; batch 32; mixed precision (AMP)
  - Cross-entropy dengan *label smoothing* 0,1 (membantu menghadapi label noise)
  - Augmentasi: RandomResizedCrop(224, skala 0,4–1,0), flip horizontal, TrivialAugmentWide, RandomErasing(p=0,25)
- **Seleksi model**: eksperimen dibandingkan pada split validasi lokal yang sama (15% per kelas, seed 42) dengan Macro-F1 dari fungsi `metrics()` di `evaluate.py` resmi. Pemilihan memakai **rata-rata Macro-F1 3 epoch terakhir**, bukan nilai maksimum. Model submission lalu dilatih ulang pada seluruh data yang tersedia (my_train + my_val = 2.753 gambar) dengan konfigurasi dan jadwal yang sama.
- **Data tambahan**: tidak ada. Hanya train set resmi, tetapi **baru 2.753 dari 2.964 gambar**: 211 gambar hilang dari salinan dataset kelompok. Kekurangan ini terdeteksi dengan mencocokkan isi folder ke `train_sources.csv`.
- **Ukuran input & preprocessing**: resize sisi pendek 256 → center crop 224 → normalisasi mean/std ImageNet. Disimpan di `submission/config.json` dan dibaca oleh `get_transform()`.
- **Test-time augmentation**: **aktif**. Prediksi = rata-rata logits gambar asli dan cerminan horizontalnya (`tta_hflip` di `config.json`). Estimasi evaluasi 740 gambar di CPU 2 core: ±3 menit.
- **Bobot**: fp16 (55,7 MB) agar di bawah batas 100 MB GitHub; argmax identik dengan fp32 (diuji).

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `0d4bd7a` | 0,9722 | 0,9595 (akurasi 0,9609) |
| Preview 2 (21 Okt) | `4daca6c` | 0,9722 (konfigurasi sama, model A) | 0,9675 (akurasi 0,969) |
| Final (25 Okt)     | `4daca6c` | — (model sama dengan Preview 2) | |

Kolom *Commit* menunjuk commit yang memuat isi `submission/` yang dievaluasi. Asisten mengevaluasi commit terakhir sebelum tenggat. Commit sesudah `4daca6c` hanya mengubah README.

Catatan eksperimen:

| ID | Konfigurasi | Macro-F1 lokal | Catatan |
|----|-------------|----------------|---------|
| E0 | Baseline ResNet-18 (starter kit, 8 epoch) | — | referensi: ~0,92 di validation set tersembunyi (halaman kompetisi) |
| E1 | ConvNeXt-Tiny 224, 15 epoch, aug strong; dilatih pada my_train (2.343 dari 2.753 gambar) | 0,9722 (epoch 12) | Model Preview 1. Kesalahan dominan: keluarga gelas (plastik/kertas/styrofoam), bentuk sama tetapi material berbeda → hipotesis: resolusi lebih tinggi membantu membedakan tekstur |
| E2 | cnxt224: konfigurasi E1, data & split sama | 0,9722 (epoch 12); rata-3-akhir 0,9711 | **Identik angka per angka dengan E1**: pipeline reproduksibel (seed 42). Terpilih |
| E3 | cnxt288: resolusi 288 px, data & split sama | 0,9735 (epoch 11); rata-3-akhir 0,9704 | Hipotesis **tidak terbukti**: selisih dengan E2 hanya ±1 gambar per kelas |
| E4 | TTA flip pada cnxt224 | 0,9722 → 0,9722 | Tidak mengubah satu pun prediksi di `my_val`. Dipakai karena tidak merugikan |
| P2 | cnxt224 dilatih ulang pada 2.753 gambar, 15 epoch, TTA flip | — (tanpa validasi) | Preview 2 → **model Final** |

### Distribusi Dataset & Split Validasi Lokal

Pencocokan isi folder `train/` terhadap `train_sources.csv` serta pembagian split lokal menggunakan `make_local_split('data/train', '.', 0.15, 42)` (fraksi 15% per kelas, `seed=42`):

| Kelas | Manifest (`train_sources.csv`) | Tersedia | Hilang | `my_train` (85%) | `my_val` (15%) |
|-------|-------------------------------:|---------:|-------:|-----------------:|---------------:|
| `botol_plastik` | 541 | 523 | 18 | 445 | 78 |
| `gelas_plastik` | 572 | 521 | 51 | 443 | 78 |
| `kertas_kardus` | 395 | 365 | 30 | 311 | 54 |
| `saset_kemasan` | 758 | 673 | 85 | 573 | 100 |
| `styrofoam` | 698 | 671 | 27 | 571 | 100 |
| **TOTAL** | **2.964** | **2.753** | **211** | **2.343** | **410** |

### Evaluasi Per Kelas & Confusion Matrix pada `my_val`

Hasil evaluasi `evaluate.py` untuk model validasi ConvNeXt-Tiny 224 px (dilatih pada 2.343 gambar `my_train`, diuji pada 410 gambar `my_val`):

| Kelas | Precision | Recall | F1-Score | Jumlah Sampel (`my_val`) |
|-------|----------:|-------:|---------:|-------------------------:|
| `botol_plastik` | 0,9512 | 1,0000 | **0,9750** | 78 |
| `gelas_plastik` | 0,9868 | 0,9615 | **0,9740** | 78 |
| `kertas_kardus` | 1,0000 | 0,9259 | **0,9615** | 54 |
| `saset_kemasan` | 0,9703 | 0,9800 | **0,9751** | 100 |
| `styrofoam` | 0,9703 | 0,9800 | **0,9751** | 100 |
| **Macro-F1 / Akurasi** | — | — | **0,9722** (Akurasi: **0,9732**) | **410** (399 benar, 11 salah) |

**Confusion Matrix** (baris = label asli, kolom = prediksi):

| Label Asli \ Prediksi | `botol_plastik` | `gelas_plastik` | `kertas_kardus` | `saset_kemasan` | `styrofoam` |
|-----------------------|----------------:|----------------:|----------------:|----------------:|------------:|
| **`botol_plastik`** | **78** | 0 | 0 | 0 | 0 |
| **`gelas_plastik`** | 1 | **75** | 0 | 0 | 2 |
| **`kertas_kardus`** | 0 | 1 | **50** | 2 | 1 |
| **`saset_kemasan`** | 2 | 0 | 0 | **98** | 0 |
| **`styrofoam`** | 1 | 0 | 0 | 1 | **98** |

Dari 11 kesalahan pada `my_val`, kelas `kertas_kardus` memiliki *recall* terendah (50/54 benar; 4 sampel terprediksi sebagai `saset_kemasan`, `gelas_plastik`, dan `styrofoam`), diikuti `gelas_plastik` yang 2 sampelnya tertukar dengan `styrofoam` dan 1 sampel dengan `botol_plastik`.

## Pelajaran

1. **Validasi lokal optimis.** Skor lokal 0,9722 turun menjadi 0,9595 di validation set tersembunyi. Penyebabnya, checkpoint dipilih dari nilai maksimum kurva yang berisik dan `my_val` hanya 410 gambar. Karena itu seleksi diganti menjadi rata-rata 3 epoch terakhir, dan skor preview dijadikan dasar pemilihan model Final.
2. **Resolusi bukan hambatan utama.** 288 px tidak lebih baik dari 224 px. Kesalahan yang tersisa didominasi gambar ambigu atau label keliru: poster daur ulang, gelas berlabel "polystyrene cups", screenshot media sosial.
3. **TTA flip netral.** Model yang dilatih dengan augmentasi flip sudah hampir invarian terhadap flip.
4. **Periksa kelengkapan data terhadap manifest.** 211 gambar hilang tidak terdeteksi di Preview 1. Sejak Preview 2, isi folder dicocokkan nama per nama dengan `train_sources.csv`.
5. **Data latih lebih banyak.** Melatih pada seluruh data yang tersedia (+410 gambar) dengan jadwal penuh menaikkan skor tersembunyi dari 0,9595 menjadi 0,9675 (+0,008; akurasi 0,9609 → 0,969, sekitar 6 gambar lebih banyak yang benar).

## Cara Reproduksi

Dataset tidak disertakan di repo (sesuai aturan kompetisi). Letakkan folder `train/` dari Google Drive asisten di luar repo.

**Lingkungan:** pelatihan dijalankan di Google Colab (GPU T4) dengan menjalankan `train.py` memakai dua perintah di bawah. Seluruh logika pelatihan ada di `train.py`; notebook Colab hanya memanggil perintah yang sama dan tidak disertakan di repo.

```bash
pip install -r requirements.txt
# eksperimen dengan validasi (my_val dibuat otomatis, seed 42)
python train.py --data ../data/train --split-dir . --name A_cnxt224 --epochs 15 --lr 0.0001 --batch-size 32 --aug strong --seed 42 --backbone convnext_tiny.fb_in22k_ft_in1k --img-size 224
# model submission: data penuh, jadwal yang sama, TTA flip
python train.py --data ../data/train --split-dir . --name B_cnxt224_full_tta --epochs 15 --lr 0.0001 --batch-size 32 --aug strong --seed 42 --backbone convnext_tiny.fb_in22k_ft_in1k --img-size 224 --full --tta-hflip
# salin isi runs/B_cnxt224_full_tta/submission/ ke submission/, lalu:
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform() — membaca config.json (termasuk tta_hflip)
  config.json       # backbone + ukuran input + mean/std + tta_hflip
  model.pth         # state_dict (fp16, 55,7 MB)
  requirements.txt  # timm==1.0.30
evaluate.py         # salinan skrip evaluasi resmi (tidak diubah)
train.py            # training: seleksi Macro-F1, log, ekspor submission, --full, --tta-hflip
train_baseline.py   # baseline dari starter kit
```
