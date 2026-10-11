# Panduan Eksperimen — Studio 2

Catatan kerja internal kelompok. README resmi tetap memakai `README_TEMPLATE.md`.

## Struktur

```
repo/
├── train.py                 # training (Macro-F1, log, ekspor submission)
├── evaluate.py              # skrip resmi — JANGAN diubah
├── submission/              # yang dinilai asisten
│   ├── model.py             # generik, tidak perlu diubah
│   ├── config.json          # backbone + preprocessing (dari run terpilih)
│   ├── model.pth            # bobot (dari run terpilih)
│   └── requirements.txt
├── tools/audit_sources.py   # cari kandidat label salah dari train_sources.csv
├── audit/suspects.csv       # hasil audit: 40 kandidat untuk dicek manual
├── notebooks/studio2_kasus_final.ipynb  # Colab: kasus -> final -> paket submission
└── runs/                    # hasil tiap eksperimen (di-.gitignore)
```

Prinsip utamanya: **satu run = satu folder submission lengkap**. `runs/<nama>/submission/` berisi `model.py`, `config.json`, dan `model.pth` yang pasti cocok satu sama lain. Folder `submission/` di root hanya diganti saat kamu *mempromosikan* run terbaik.

### Transisi dari baseline

`submission/config.json` bawaan berisi preset `resnet18`. Dengan preset ini, `model.pth` hasil `train_baseline.py` **tetap bisa dinilai** dengan `model.py` yang baru (sudah diuji: bobot ResNet-18 torchvision termuat `strict=True` ke ResNet-18 timm, outputnya identik). Jadi submission baseline-mu tetap aman sambil menunggu hasil eksperimen.

## 1. Training

**Colab (disarankan, GPU T4):** upload `notebooks/studio2_kasus_final.ipynb`, isi `DATA_SRC`, lalu *Run all*. Notebook ini mandiri (kode sudah tertanam) dan menghasilkan dua paket submission siap pakai.

**Lokal:**
```bash
python train.py --data ../data/train --backbone convnext_tiny.fb_in22k_ft_in1k --epochs 15 --name cnxt_t_e15
```

Split `my_val` dibuat dengan seed 42 dan **identik** di mesin mana pun, sama dengan split `train_baseline.py`. Jadi skor dari Colab dan dari laptop bisa dibandingkan langsung.

Kalau di laptop belum ada `my_val/`, buat split-nya saja tanpa training:
```bash
python -c "from train import make_local_split; make_local_split('../data/train', '.', 0.15, 42)"
```

## 2. Cek dan promosikan run terbaik

Dari Colab, ekstrak zip yang terunduh ke folder `runs/` di repo, lalu:

```powershell
python evaluate.py --submission runs/cnxt_t_e15/submission --data ./my_val
Copy-Item runs\cnxt_t_e15\submission\model.pth, runs\cnxt_t_e15\submission\config.json submission\ -Force
python evaluate.py --submission ./submission --data ./my_val    # harus sama persis
```

Setelah promosi, buka `submission/config.json`, lihat `timm_version`, lalu ganti isi `submission/requirements.txt` menjadi versi itu (contoh: `timm==1.0.30`). Tujuannya agar environment asisten sama dengan environment training.

## 3. Commit

```bash
git add --dry-run .     # pastikan tidak ada .jpg dan tidak ada runs/
git add .
git commit -m "E1: convnext_tiny 15 epoch, macroF1 lokal 0.xx"
git push
```

Tulis skor dan nama run di pesan commit. Riwayat commit dinilai sebagai jejak progres.

**Uji dari clone bersih** sebelum setiap tenggat:
```bash
git clone <url-repo> ../cek-clone
python ../cek-clone/evaluate.py --submission ../cek-clone/submission --data ./my_val
```

## 4. Rencana eksperimen

Aturan: **ubah satu hal per run**. Kalau dua hal diubah sekaligus, kamu tidak tahu mana yang berpengaruh.

`my_val` hanya ~440 gambar, jadi selisih Macro-F1 di bawah ~0,01 masih dalam batas noise. Kalau dua konfigurasi berbeda tipis, ulangi dengan `--seed` lain sebelum menyimpulkan.

| ID | Perubahan dari E1 | Argumen | Hipotesis |
|----|-------------------|---------|-----------|
| E0 | baseline ResNet-18 | `train_baseline.py` | referensi (~0,92 di val tersembunyi) |
| E1 | ConvNeXt-Tiny in22k | default | pretrained lebih kuat → +F1 |
| E2 | backbone lebih ringan | `--backbone efficientnet_b2.ra_in1k` | lebih cepat; F1 sedikit lebih rendah? |
| E3 | augmentasi ringan | `--aug light` | strong lebih tahan noise/variasi web? |
| E4 | resolusi lebih besar | `--img-size 288` | detail leher botol/tekstur lebih jelas |
| E5 | LR | `--lr 5e-5` dan `--lr 2e-4` | cari LR yang stabil |
| E6 | epoch lebih panjang | `--epochs 25` | lihat di `log.csv` apakah masih naik |

Catat setiap run di tabel ini: ID, nama run, Macro-F1 lokal, F1 terendah (kelas apa), catatan.

## 5. Analisis error dan label noise

- `runs/<nama>/val_predictions.csv` diurutkan dari prediksi yang paling "yakin salah". Buka 20 gambar teratas, lalu tanyakan untuk setiap gambar: **model yang salah, atau labelnya yang salah?**
- `audit/suspects.csv` berisi 40 kandidat dari judul halaman sumber. Kategori yang paling meyakinkan: 10 kemasan kertas yang dilabeli `saset_kemasan`, 2 gelas kertas yang dilabeli `gelas_plastik`, dan 1 gelas plastik yang dilabeli `styrofoam`. Sisanya (26) kemungkinan foto tumpukan sampah campuran.
- Perbaiki **hanya yang jelas salah** setelah melihat gambarnya. Validation set tersembunyi juga mengandung noise, jadi jangan mengejar data latih yang "sempurna".
- Penting: perbaikan label sebaiknya dilakukan **di `train/` asli sebelum membuat split**. Jika `my_val/` sudah ada, hapus `my_train/` dan `my_val/`, perbaiki, lalu split ulang. Akibatnya skor sebelum dan sesudah perbaikan tidak bisa dibandingkan langsung.

## 6. Latihan mandiri

1. **Crop vs squash.** `build_eval_transform` memotong tepi gambar (CenterCrop). Hipotesis: leher botol, pembeda utama botol vs gelas, kadang ikut terpotong. Buat varian yang me-resize langsung ke `(img_size, img_size)` tanpa crop, lalu bandingkan F1 `botol_plastik` dan `gelas_plastik`. Petunjuk: transform train dan eval harus konsisten. Pikirkan bagian mana di `build_train_transform` yang juga perlu diubah.
2. **Kelas minoritas.** Kalau F1 `kertas_kardus` paling rendah, coba `WeightedRandomSampler`. Ukur dulu sebelum menambah kompleksitas.
3. **TTA flip (untuk Final).** Rata-ratakan logits gambar asli dan hasil flip horizontal di dalam `get_model()`, tanpa mengubah nama key `state_dict`. Bagaimana caranya?

## 7. Checklist sebelum Final (target selesai Sabtu, 24 Okt)

- [ ] Konfigurasi terbaik dilatih ulang dengan `--full` dan `--epochs` = epoch terbaik dari `log.csv`
- [ ] `submission/` berisi model.py + config.json + model.pth (< 100 MB)
- [ ] `requirements.txt` di-pin sesuai `timm_version`
- [ ] README: nama kelompok, nama lengkap + NIM semua anggota, pendekatan, cara reproduksi
- [ ] Uji clone bersih berhasil
- [ ] Repo publik, commit sudah di-push sebelum 23:59 WIB
