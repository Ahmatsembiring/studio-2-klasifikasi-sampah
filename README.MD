# Studio 2 — Klasifikasi Sampah · KELOMPOK 1

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Ahmat Prayoga Sembiring | 123140053 |  |
| 2  | Tengku Hafid Diraputra | 123140043 |  |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny pretrained ImageNet-22k → fine-tuned ImageNet-1k (`timm`: `convnext_tiny.fb_in22k_ft_in1k`), head linear 5 kelas. Seluruh layer di-*fine-tune* (tanpa freezing).
- **Strategi pelatihan**:
  - AdamW, LR backbone 0.0001, LR head 10×, weight decay 0,05; warmup linear 1 epoch lalu cosine decay per-step; 15 epoch; batch 32; mixed precision (AMP)
  - Cross-entropy dengan *label smoothing* 0,1 (membantu menghadapi label noise)
  - Augmentasi: RandomResizedCrop(224, skala 0,4–1,0), flip horizontal, TrivialAugmentWide, RandomErasing(p=0,25)
- **Seleksi model**: eksperimen dibandingkan pada split validasi lokal yang sama (15% per kelas, seed 42) dengan Macro-F1 dari fungsi `metrics()` di `evaluate.py` resmi. Pemilihan memakai **rata-rata Macro-F1 3 epoch terakhir**, bukan nilai maksimum, untuk menghindari bias seleksi pada validasi yang kecil. Model submission lalu dilatih ulang pada seluruh data yang tersedia (my_train + my_val = 2.753 gambar) dengan konfigurasi dan jadwal yang sama.
- **Data tambahan**: tidak ada. Hanya train set resmi, tetapi **baru 2.753 dari 2.964 gambar**: 211 gambar hilang dari salinan dataset kelompok. Kekurangan ini terdeteksi dengan mencocokkan isi folder ke `train_sources.csv`. Pelatihan ulang dengan data lengkap direncanakan.
- **Ukuran input & preprocessing**: resize sisi pendek 256 → center crop 224 → normalisasi mean/std ImageNet. Disimpan di `submission/config.json` dan dibaca oleh `get_transform()`.
- **Test-time augmentation**: **aktif**. Prediksi = rata-rata logits gambar asli dan cerminan horizontalnya (`tta_hflip` di `config.json`). Di `my_val`: 0,9722 tanpa TTA → 0,9722 dengan TTA (tidak mengubah prediksi; tetap dipakai karena tidak merugikan). Estimasi evaluasi 740 gambar di CPU 2 core: ±3,6 menit.
- **Bobot**: fp16 (55,7 MB) agar di bawah batas 100 MB GitHub; argmax identik dengan fp32 (diuji).

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `f88e1ad` | 0,9722 | 0,9595 (akurasi 0,9609) |
| Preview 2 (21 Okt) | `<isi hash setelah push>` | 0,9722 (cnxt224 + TTA, model A) | |
| Final (25 Okt)     | | | |

Validasi lokal terbukti **optimis**: 0,9722 di lokal vs 0,9595 di validation set tersembunyi (selisih 0,013). Penyebabnya, checkpoint Preview 1 dipilih dari nilai *maksimum* kurva yang berisik, dan `my_val` kecil (410 gambar). Karena itu, mulai Preview 2 pemilihan memakai rata-rata 3 epoch terakhir.

Catatan eksperimen:

| ID | Konfigurasi | Macro-F1 lokal | Catatan |
|----|-------------|----------------|---------|
| E0 | Baseline ResNet-18 (starter kit, 8 epoch) | — | referensi: ~0,92 di validation set tersembunyi (halaman kompetisi) |
| E1 | ConvNeXt-Tiny 224, 15 epoch, aug strong, 2.753 gambar | 0,9722 (epoch 12) | Model Preview 1. Kesalahan dominan: keluarga gelas (plastik/kertas/styrofoam), bentuk sama tetapi material berbeda → hipotesis: resolusi lebih tinggi membantu membedakan tekstur |
| E2 | cnxt224: konfigurasi E1, data & split sama | 0,9722 (epoch 12); rata-3-akhir 0,9711 | **Identik angka per angka dengan E1**: pipeline reproduksibel (seed 42). F1 gelas 0,974, kertas 0,962, styrofoam 0,975. **Terpilih** |
| E3 | cnxt288: resolusi 288 px, data & split sama | 0,9735 (epoch 11); rata-3-akhir 0,9704 | Hipotesis **tidak terbukti**: selisih dengan E2 hanya ±1 gambar per kelas (gelas 0,967, kertas 0,952, styrofoam 0,980). Kesalahan tersisa didominasi gambar ambigu atau label keliru (poster daur ulang, gelas berlabel "polystyrene cups", screenshot), bukan kurangnya detail piksel |
| E4 | TTA flip pada cnxt224 | 0,9722 → 0,9722 | Tidak mengubah satu pun prediksi di `my_val`: model sudah hampir invarian flip karena dilatih dengan flip. Dipakai karena tidak merugikan |
| P2 | cnxt224 dilatih ulang pada seluruh data yang tersedia (2.753 gambar), 15 epoch, TTA flip | — (tanpa validasi; konfigurasi sama dengan E2) | Dikirim untuk Preview 2. Dibanding model Preview 1: +410 gambar latih, jadwal penuh, TTA |

## Cara Reproduksi

Dataset tidak disertakan di repo (sesuai aturan kompetisi). Letakkan folder `train/` dari Google Drive asisten di luar repo.

**Google Colab (yang dipakai kelompok):** buka `notebooks/studio2_preview2.ipynb`, pilih runtime T4 GPU, isi `DATA_SRC`, lalu *Run all*. Notebook menjalankan eksperimen, uji TTA, model final, dan membuat paket submission.

**Lokal (setara dengan notebook):**

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
notebooks/          # notebook Colab versi bersih (tanpa output)
```
