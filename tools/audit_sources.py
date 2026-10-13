"""Audit label dari train_sources.csv — cari gambar yang LABEL-nya patut dicurigai.

Idenya: judul halaman sumber (kolom `title`) sering menyebut jenis bendanya.
Kalau label = gelas_plastik tapi judulnya "Paper Cup ...", gambar itu layak dicek.
Ini hanya PENYARING kandidat — keputusan akhir tetap dengan MELIHAT gambarnya.

    python tools/audit_sources.py --csv ../data/train_sources.csv --out suspects.csv

Output: suspects.csv (file, label, saran_label, alasan, title, image_url)
"""
import argparse

import pandas as pd

# (label sekarang, regex yang HARUS ada di judul, regex yang TIDAK boleh ada, saran label, alasan)
RULES = [
    ("gelas_plastik", r"paper|papercup|kertas|kraft", r"plasti|\bpet\b",
     "kertas_kardus", "gelas kertas berlabel gelas_plastik"),
    ("saset_kemasan", r"paper (?:box|bag|wrapper|sandwich|noodle)|parchment|kertas bungkus|noodle box|kraft paper handle",
     r"plasti|sachet|saset", "kertas_kardus", "kemasan kertas berlabel saset_kemasan"),
    ("styrofoam", r"plastic cups?|clear plastic|gelas plastik", r"styro|foam|gabus|polystyrene",
     "gelas_plastik", "gelas plastik berlabel styrofoam"),
    ("kertas_kardus", r"plastik|plastic", r"paper|kertas|kardus|cardboard|karton",
     "?", "judul plastik tanpa kata kertas"),
]
# Judul generik (berita/kampanye) -> kemungkinan foto tumpukan sampah campuran.
SCENE = r"\bban\b|banned|pollution|polusi|landfill|sampah plastik|plastic waste|recycling company|single-use"
OWN = {
    "botol_plastik": r"botol|bottle",
    "gelas_plastik": r"gelas|cup",
    "kertas_kardus": r"kertas|kardus|karton|paper|cardboard",
    "saset_kemasan": r"sa[sc]h?et|pouch|bungkus|kemasan|wrapper|mi |mie|noodle|snack",
    "styrofoam": r"styro|sterofoam|stirofoam|foam|gabus|polystyrene",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="suspects.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    t = df["title"].str.lower()
    rows = []
    for label, must, must_not, sugg, why in RULES:
        m = (df.label == label) & t.str.contains(must, regex=True) & ~t.str.contains(must_not, regex=True)
        rows.append(df[m].assign(saran_label=sugg, alasan=why))
    for label, own in OWN.items():
        m = (df.label == label) & t.str.contains(SCENE, regex=True) & ~t.str.contains(own, regex=True)
        rows.append(df[m].assign(saran_label="?", alasan="kemungkinan foto sampah campuran"))
    multi = df.groupby("page_url").label.transform("nunique") > 1
    rows.append(df[multi].assign(saran_label="?", alasan="satu halaman, label berbeda"))

    out = (pd.concat(rows).drop_duplicates("file")
           [["file", "label", "saran_label", "alasan", "title", "image_url"]]
           .sort_values(["alasan", "file"]))
    out.to_csv(a.out, index=False)
    print(f"{len(out)} kandidat -> {a.out}")
    print(out.groupby("alasan").size().to_string())


if __name__ == "__main__":
    main()
