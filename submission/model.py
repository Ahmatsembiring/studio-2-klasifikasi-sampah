"""Submission Studio 2 — model generik berbasis timm.

Arsitektur & preprocessing TIDAK ditulis langsung di sini, tetapi dibaca dari
`config.json` di folder yang sama (dibuat otomatis oleh train.py). Alasannya:
konfigurasi selalu "ikut" bersama bobotnya, sehingga model.py, config.json,
dan model.pth tidak mungkin saling tidak cocok tanpa ketahuan.

Isi folder submission/ yang wajib di-commit:
    model.py      (file ini — tidak perlu diubah)
    config.json   (backbone, ukuran input, mean/std, tta_hflip)
    model.pth     (state_dict)

Kontrak evaluate.py:
    get_model()     -> nn.Module, output logits [batch, 5], TANPA mengunduh bobot
    get_transform() -> PIL.Image (RGB) -> Tensor [3, H, W]
Urutan kelas: 0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
"""
import json
from pathlib import Path

import timm
import torch
from torchvision import transforms

NUM_CLASSES = 5
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path=CONFIG_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"config.json tidak ditemukan di {Path(path).parent} — "
                                "salin config.json dari folder run bersama model.pth.")
    return json.loads(Path(path).read_text())


def build_model(backbone, pretrained=False):
    """Dipakai train.py (pretrained=True) dan get_model() (pretrained=False).
    Fungsi yang sama -> nama key state_dict pasti identik."""
    return timm.create_model(backbone, pretrained=pretrained, num_classes=NUM_CLASSES)


def build_eval_transform(img_size, crop_pct, mean, std):
    return transforms.Compose([
        transforms.Resize(int(round(img_size / crop_pct))),  # sisi pendek
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def add_hflip_tta(model):
    """Test-time augmentation: rata-rata logits gambar asli dan cerminan horizontalnya.

    Hanya `forward` yang dibungkus — tidak ada layer baru, jadi nama key state_dict
    tetap sama dan load_state_dict(strict=True) di evaluate.py tetap lolos.
    Masuk akal karena model dilatih dengan RandomHorizontalFlip: kedua versi gambar
    sama-sama "normal" bagi model, dan merata-ratakannya mengurangi variansi prediksi.
    """
    base_forward = model.forward

    def forward(x):
        return (base_forward(x) + base_forward(torch.flip(x, dims=[3]))) / 2

    model.forward = forward
    return model


def get_model():
    cfg = load_config()
    model = build_model(cfg["backbone"], pretrained=False)
    if cfg.get("tta_hflip", False):   # config lama tanpa kunci ini -> tanpa TTA
        model = add_hflip_tta(model)
    return model


def get_transform():
    cfg = load_config()
    return build_eval_transform(cfg["img_size"], cfg["crop_pct"], cfg["mean"], cfg["std"])
