"""Training Studio 2 — fine-tuning backbone timm, checkpoint dipilih berdasarkan Macro-F1.

Jalankan dari root repo:
    python train.py --data ../data/train --backbone convnext_tiny.fb_in22k_ft_in1k --epochs 15 --name cnxt_e15
    python evaluate.py --submission runs/cnxt_e15/submission --data ./my_val

Setiap run menghasilkan runs/<name>/:
    submission/          model.py + config.json + model.pth -> folder submission lengkap
    log.csv              metrik per epoch (loss, Macro-F1, akurasi, F1 per kelas)
    val_predictions.csv  prediksi checkpoint terbaik di my_val, diurutkan dari yang paling "salah"
    args.json            semua argumen run (reproduksibilitas)

Mode --full: latih di my_train + my_val dengan jumlah epoch tetap (untuk model Final).
--tta-hflip: config.json submission menyalakan TTA flip horizontal (lihat submission/model.py).

Perbedaan utama dari train_baseline.py:
  1. Checkpoint dipilih berdasarkan Macro-F1 (metrik penilaian), bukan akurasi.
  2. Metrik & jalur load memakai fungsi dari evaluate.py -> angka identik dengan penilai.
  3. Mean/std/ukuran input diambil dari pretrained_cfg backbone, disimpan di config.json.
  4. Warmup + cosine per-step, LR head lebih besar dari backbone, AMP di GPU.
  5. Bobot > 95 MB otomatis disimpan fp16 agar muat di GitHub (< 100 MB).
"""
import argparse
import contextlib
import csv
import json
import math
import random
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms

import evaluate                       # metrik + jalur load yang SAMA dengan penilai
from submission import model as sub   # build_model / build_eval_transform

# GradScaler melewati optimizer.step() di batch pertama (cek overflow) -> peringatan scheduler tidak relevan.
warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step\(\).*")

CLASSES = evaluate.CLASSES
FP16_EXPORT_MB = 95


# ----------------------------------------------------------------------------- data
def make_local_split(src, dst, frac, seed):
    """Sama persis dengan train_baseline.py -> split my_val identik bila seed/frac sama.
    Hanya dibuat sekali; kalau my_val/ sudah ada, split lama dipakai ulang."""
    dst_tr, dst_va = Path(dst) / "my_train", Path(dst) / "my_val"
    if dst_va.exists():
        return dst_tr, dst_va
    rng = random.Random(seed)
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.jpg"))
        rng.shuffle(files)
        n_va = int(len(files) * frac)
        for i, f in enumerate(files):
            out = (dst_va if i < n_va else dst_tr) / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
    return dst_tr, dst_va


def build_train_transform(cfg, aug):
    size = cfg["img_size"]
    to_tensor = [transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]
    if aug == "light":  # setara baseline
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            *to_tensor,
        ])
    return transforms.Compose([  # "strong"
        transforms.RandomResizedCrop(size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        *to_tensor,
        transforms.RandomErasing(p=0.25),
    ])


# ----------------------------------------------------------------------------- utils
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_config(model, img_size):
    pc = model.pretrained_cfg
    return {
        "backbone": None,  # diisi pemanggil
        "img_size": img_size or pc["input_size"][-1],
        "crop_pct": pc.get("crop_pct") or 0.875,
        "mean": [float(v) for v in pc["mean"]],
        "std": [float(v) for v in pc["std"]],
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
    }


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    probs, ys, loss_sum = [], [], 0.0
    for x, y in dl:
        logits = model(x.to(device)).float()
        loss_sum += F.cross_entropy(logits, y.to(device), reduction="sum").item()
        probs.append(logits.softmax(1).cpu())
        ys.append(y)
    p, y = torch.cat(probs), torch.cat(ys)
    return y.tolist(), p, loss_sum / len(y)


def export_submission(state, cfg, out_dir):
    """Tulis folder submission lengkap: model.py + config.json + model.pth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(sub.__file__), out_dir / "model.py")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    mb = sum(v.numel() * v.element_size() for v in state.values()) / 1e6
    if mb > FP16_EXPORT_MB:
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
        print(f"bobot {mb:.0f} MB > {FP16_EXPORT_MB} MB -> disimpan fp16")
    torch.save(state, out_dir / "model.pth")
    return (out_dir / "model.pth").stat().st_size / 1e6


@torch.no_grad()
def official_check(sub_dir, data_dir, device):
    """Muat submission PERSIS seperti evaluate.py (strict=True, weights_only=True)."""
    model, tf = evaluate.load_submission(sub_dir, device)
    items = evaluate.list_images(data_dir)
    y_true, y_pred = [], []
    for i in range(0, len(items), 32):
        chunk = items[i:i + 32]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in chunk]).to(device)
        y_pred += model(x).argmax(1).cpu().tolist()
        y_true += [c for _, c in chunk]
    return evaluate.metrics(y_true, y_pred)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder train/ asli dari dataset")
    ap.add_argument("--split-dir", default=".", help="lokasi my_train/ dan my_val/")
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k", help="nama model timm")
    ap.add_argument("--img-size", type=int, default=None, help="default: ukuran pretrained backbone")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="LR backbone")
    ap.add_argument("--head-lr-mult", type=float, default=10.0, help="LR head = lr x nilai ini")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--aug", choices=["light", "strong"], default="strong")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default=None, help="nama folder di runs/")
    ap.add_argument("--full", action="store_true", help="latih di my_train + my_val (model Final)")
    ap.add_argument("--no-pretrained", action="store_true", help="tanpa bobot ImageNet (uji/scratch)")
    ap.add_argument("--tta-hflip", action="store_true",
                    help="aktifkan TTA flip horizontal di config.json submission (berlaku saat evaluasi)")
    a = ap.parse_args()

    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    name = a.name or f"{a.backbone.split('.')[0]}_{time.strftime('%m%d-%H%M')}"
    run = Path("runs") / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=2))

    # ---- model + konfigurasi preprocessing (sumber kebenaran tunggal)
    model = sub.build_model(a.backbone, pretrained=not a.no_pretrained)
    cfg = model_config(model, a.img_size)
    cfg["backbone"] = a.backbone
    cfg["tta_hflip"] = a.tta_hflip
    model.to(device)
    print(f"device={device}  backbone={a.backbone}  input={cfg['img_size']}  mean={cfg['mean']}")

    # ---- data
    tr_dir, va_dir = make_local_split(a.data, a.split_dir, 0.15, 42)
    train_tf = build_train_transform(cfg, a.aug)
    eval_tf = sub.build_eval_transform(cfg["img_size"], cfg["crop_pct"], cfg["mean"], cfg["std"])
    tr_ds = datasets.ImageFolder(tr_dir, train_tf)
    va_ds = datasets.ImageFolder(va_dir, eval_tf)
    assert tr_ds.classes == CLASSES == va_ds.classes, f"urutan kelas salah: {tr_ds.classes}"
    if a.full:
        tr_ds = ConcatDataset([tr_ds, datasets.ImageFolder(va_dir, train_tf)])
    print(f"train={len(tr_ds)}  val={len(va_ds)}  mode={'FULL (tanpa validasi)' if a.full else 'normal'}")

    kw = dict(num_workers=a.workers, pin_memory=device == "cuda", persistent_workers=a.workers > 0)
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True, drop_last=True, **kw)
    va_dl = DataLoader(va_ds, a.batch_size, shuffle=False, **kw)

    # ---- optimizer: head belajar lebih cepat daripada backbone pretrained
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": a.lr},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": a.lr * a.head_lr_mult},
    ], weight_decay=a.wd)
    total = a.epochs * len(tr_dl)
    warm = max(1, int(a.warmup_epochs * len(tr_dl)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else contextlib.nullcontext

    # ---- training loop
    header = ["epoch", "lr", "train_loss", "val_loss", "macro_f1", "acc"] + [f"f1_{c}" for c in CLASSES] + ["sec"]
    log_f = open(run / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(header)
    best_f1, best_loss, best_ep, best_state, best_probs = -1.0, float("inf"), 0, None, None

    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with amp_ctx():
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(y)
            n += len(y)

        y_true, probs, val_loss = predict(model, va_dl, device)
        m = evaluate.metrics(y_true, probs.argmax(1).tolist())
        sec = time.time() - t0
        log.writerow([ep, f"{opt.param_groups[0]['lr']:.2e}", f"{loss_sum / n:.4f}", f"{val_loss:.4f}",
                      f"{m['macro_f1']:.4f}", f"{m['accuracy']:.4f}"]
                     + [f"{m['per_class_f1'][c]:.4f}" for c in CLASSES] + [f"{sec:.0f}"])
        log_f.flush()

        # Macro-F1 lebih tinggi menang; bila seri, pilih val_loss lebih rendah.
        # Mode --full tidak punya validasi yang jujur -> selalu ambil epoch terakhir.
        improved = a.full or m["macro_f1"] > best_f1 or (m["macro_f1"] == best_f1 and val_loss < best_loss)
        if improved:
            best_f1, best_loss, best_ep, best_probs = m["macro_f1"], val_loss, ep, probs
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:2d}/{a.epochs}  train_loss={loss_sum / n:.4f}  val_loss={val_loss:.4f}  "
              f"macroF1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  {sec:.0f}s {'*' if improved else ''}")
    log_f.close()

    # ---- prediksi checkpoint terbaik -> bahan analisis error & label noise
    with open(run / "val_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "pred", "p_label", "p_pred", "benar"])
        rows = []
        for (path, lab), p in zip(va_ds.samples, best_probs):
            pr = int(p.argmax())
            rows.append([Path(path).relative_to(va_dir).as_posix(), CLASSES[lab], CLASSES[pr],
                         round(float(p[lab]), 4), round(float(p[pr]), 4), int(pr == lab)])
        w.writerows(sorted(rows, key=lambda r: r[3]))  # paling "yakin salah" di atas

    # ---- ekspor + cek lewat jalur evaluate.py
    sub_dir = run / "submission"
    size_mb = export_submission(best_state, cfg, sub_dir)
    check = official_check(sub_dir, va_dir, device)
    (run / "best.json").write_text(json.dumps({
        "epoch": best_ep, "epochs": a.epochs, "full": a.full,
        "macro_f1_train_loop": round(best_f1, 4), "macro_f1_evaluate": round(check["macro_f1"], 4),
        "accuracy_evaluate": round(check["accuracy"], 4), "per_class_f1": check["per_class_f1"],
        "confusion_matrix": check["confusion_matrix"], "model_mb": round(size_mb, 1),
        "img_size": cfg["img_size"], "tta_hflip": a.tta_hflip, "n_train": len(tr_ds),
    }, indent=2))
    print("\n" + "=" * 70)
    print(f"run              : {run}")
    print(f"checkpoint       : epoch {best_ep}  (Macro-F1 my_val saat training = {best_f1:.4f})")
    print(f"cek evaluate.py  : Macro-F1 = {check['macro_f1']:.4f}  akurasi = {check['accuracy']:.4f}")
    print(f"model.pth        : {size_mb:.1f} MB  |  timm {timm.__version__}")
    if a.full:
        print("CATATAN: mode --full -> my_val ikut dilatih, skor di atas hanya cek 'bisa jalan'.")
    print(f"\nPromosikan ke submission/:  copy {sub_dir}/* -> submission/")
    print("=" * 70)


if __name__ == "__main__":
    main()
