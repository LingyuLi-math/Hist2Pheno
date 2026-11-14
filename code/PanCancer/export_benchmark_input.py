#!/usr/bin/env python3
"""Export the seven-cohort Hist2Pheno training input for an external benchmark.

Writes, under ``data/Input_data``:

* ``HEimgs/<dataset>/<sample>.tif`` — hardlink to the H&E whose pixels match
  ``obsm['spatial_HE']`` (no second copy of the ~84 GB of TIFFs)
* ``metadata/<dataset>/<sample>.csv`` — one row per nucleus kept on the
  pan-cancer ontology (L1 5 / L12 8 / L2 20)
* ``slides.csv``, ``ontology_labels.csv``, ``label_counts.csv``, ``README.md``

Only sections in ``data/PanCancer/section_index.csv`` with ``ok`` are exported.
That is the same 896 labeled sections used for joint training. Nuclei whose
native label is outside ``celltype_pancancer`` are omitted; their count is
``n_raw - n_nuclei`` on each slide row.

  conda activate SeededNTM
  python -u code/PanCancer/export_benchmark_input.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO / "code") not in sys.path:
    sys.path.insert(0, str(_REPO / "code"))

from Hist2Pheno_pkg.pancancer_labels import (  # noqa: E402
    load_pancancer_ontology,
    map_native_series,
)

L1_ORDER = ["Epithelial", "Immune", "Stromal", "Endothelial", "Neural"]
L12_ORDER = [
    "Tumor",
    "Epithelial",
    "T_cell",
    "B_Plasma",
    "Myeloid",
    "Stromal",
    "Endothelial",
    "Neural",
]

INPUT_ROOT = _REPO / "data" / "Input_data"
INDEX_CSV = _REPO / "data" / "PanCancer" / "section_index.csv"

# Folder names are the names to hand to an external author.
FOLDER = {
    "Lung": "lung",
    "BRCA": "brca",
    "CRC": "crc",
    "HCC": "hcc",
    "PDAC": "pdac",
    "GIST": "gist",
    "GBM": "gbm",
}

# Physical size of the linked TIFF, in µm per pixel. These are the values the
# Hist2Pheno feature extractor used. GBM microscope TIFFs were not rescaled and
# their dpi tag is not a physical pixel size, so um_per_px is left blank.
UM_PER_PX = {
    "Lung": (0.2125, "Xenium morphology, 0.2125 µm/px"),
    "BRCA": (0.42, "working HE crop; UNI scale 0.84 onto 0.5 µm/px"),
    "CRC": (0.274, "add-on H&E OME, ~0.274 µm/px"),
    "HCC": (0.5000061356695383, "CODEX H&E, treated as 0.5 µm/px"),
    "PDAC": (0.5000061356695383, "CODEX H&E, treated as 0.5 µm/px"),
    "GIST": (0.5000061356695383, "CODEX H&E, treated as 0.5 µm/px"),
    "GBM": (np.nan, "microscope H&E, native pixels; not rescaled"),
}


def he_source(dataset: str, sample: str) -> Path:
    if dataset == "Lung":
        folder = (
            _REPO
            / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases"
            / sample
        )
        hits = sorted(folder.glob("*-HE.tif"))
        if len(hits) != 1:
            raise FileNotFoundError(f"{sample}: expected one *-HE.tif, found {hits}")
        return hits[0]
    if dataset == "BRCA":
        prefix = {
            "rep1": "Xenium_FFPE_Human_Breast_Cancer_Rep1",
            "rep2": "Xenium_FFPE_Human_Breast_Cancer_Rep2",
        }[sample]
        return _REPO / f"data/Xemium/BRCA/human_breast_Xenium_{sample}/{prefix}_he_image.tif"
    if dataset == "CRC":
        folder = sample.replace("CRC", "_CRC")
        prefix = f"Xenium_V1_Human_Colon_Cancer_{folder}_Add_on_FFPE"
        return _REPO / f"data/Xemium/CRC/HE_images/{prefix}_he_image.ome.tif"
    if dataset == "HCC":
        return (
            _REPO
            / "data/CODEX/HCC/Michael_data_transfer/s4769/HE"
            / sample
            / "figures"
            / f"{sample}.tif"
        )
    if dataset in ("PDAC", "GIST"):
        return (
            _REPO
            / "data/CODEX/HCC/Michael_data_transfer/s1167"
            / sample
            / "figures"
            / f"{sample}.tif"
        )
    if dataset == "GBM":
        key = {"P174511_Initial": "174511-2-3", "P179161_Recurrent": "179161-4-3"}[sample]
        return _REPO / f"data/CODEX/GBM/WangLab/4_Images/Microscope_Image/{key}.tif"
    raise KeyError(dataset)


def _close_backed(adata) -> None:
    fh = getattr(adata, "file", None)
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


def image_hw(path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = series.shape
        axes = series.axes
    if axes == "YXS":
        return int(shape[0]), int(shape[1])
    if axes == "YX":
        return int(shape[0]), int(shape[1])
    raise ValueError(f"{path} axes {axes} shape {shape}")


def hardlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        src_stat = src.stat()
        dst_stat = dst.stat()
        if dst_stat.st_ino == src_stat.st_ino and dst_stat.st_dev == src_stat.st_dev:
            return
        dst.unlink()
    os.link(src, dst)


def export_section(row: pd.Series, onto_names: set[str]) -> dict:
    import anndata as ad

    dataset = str(row["dataset"])
    sample = str(row["sample"])
    src = he_source(dataset, sample)
    if not src.is_file():
        raise FileNotFoundError(src)
    height, width = image_hw(src)
    folder = FOLDER[dataset]
    suffix = "".join(src.suffixes) if src.name.endswith(".ome.tif") else src.suffix
    he_name = f"{sample}{suffix}"
    he_rel = Path("HEimgs") / folder / he_name
    meta_rel = Path("metadata") / folder / f"{sample}.csv"
    hardlink(src, INPUT_ROOT / he_rel)

    adata = ad.read_h5ad(row["h5ad"], backed="r")
    try:
        if "final_CT" in adata.obs.columns:
            native = adata.obs["final_CT"]
        elif "celltype" in adata.obs.columns:
            native = adata.obs["celltype"]
        else:
            raise KeyError(f"{sample} obs columns {list(adata.obs.columns)}")
        if "spatial_HE" not in adata.obsm:
            raise KeyError(f"{sample} has no obsm['spatial_HE']")
        mapped = map_native_series(native)
        keep = mapped.notna().to_numpy()
        xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)[keep]
        triples = mapped.to_numpy()[keep]
        cell_id = adata.obs.index.to_numpy()[keep].astype(str)
        native_kept = native.astype(object).to_numpy()[keep]
    finally:
        _close_backed(adata)

    if xy.shape[0] == 0:
        raise RuntimeError(f"{dataset}/{sample} kept 0 nuclei")
    oob = (
        (xy[:, 0] < -1)
        | (xy[:, 0] > width)
        | (xy[:, 1] < -1)
        | (xy[:, 1] > height)
    )
    n_oob = int(oob.sum())
    if n_oob > max(10, 0.01 * xy.shape[0]):
        raise RuntimeError(
            f"{dataset}/{sample}: {n_oob}/{xy.shape[0]} centroids fall outside "
            f"{src.name} ({width} x {height})"
        )
    level1 = [t[0] for t in triples]
    level12 = [t[1] for t in triples]
    level2 = [t[2] for t in triples]
    unknown = sorted(set(level2) - onto_names)
    if unknown:
        raise RuntimeError(f"{dataset}/{sample} L2 outside ontology: {unknown}")

    out = pd.DataFrame(
        {
            "cell_id": cell_id,
            "x_px": np.round(xy[:, 0], 4),
            "y_px": np.round(xy[:, 1], 4),
            "level1": level1,
            "level12": level12,
            "level2": level2,
            "native_label": native_kept.astype(str),
        }
    )
    dest = INPUT_ROOT / meta_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    count_rows = []
    for level in ("level1", "level12", "level2"):
        for name, k in out[level].value_counts().items():
            count_rows.append(
                {
                    "dataset": folder,
                    "sample": sample,
                    "level": level,
                    "label": name,
                    "n": int(k),
                }
            )

    um, um_note = UM_PER_PX[dataset]
    return {
        "dataset": folder,
        "cohort": dataset,
        "tech": row["tech"],
        "sample": sample,
        "he_file": str(he_rel),
        "metadata_file": str(meta_rel),
        "n_nuclei": int(len(out)),
        "n_raw_in_h5ad": int(row["n_raw"]),
        "n_keep_in_index": int(row["n_keep"]),
        "n_outside_image": n_oob,
        "image_height": height,
        "image_width": width,
        "um_per_px": um,
        "um_per_px_note": um_note,
        "source_he": str(src.relative_to(_REPO)),
        "_counts": count_rows,
    }


def write_readme(slides: pd.DataFrame, ontology: pd.DataFrame) -> None:
    lines = []
    for folder, g in slides.groupby("dataset", sort=False, observed=True):
        um = g["um_per_px"].iloc[0]
        um_txt = "" if pd.isna(um) else f"{float(um):.4g}"
        lines.append(
            f"| {folder} | {g['tech'].iloc[0]} | {len(g)} | {int(g['n_nuclei'].sum()):,} | {um_txt} |"
        )
    table = "\n".join(lines)
    l1 = " · ".join(L1_ORDER)
    l12 = " · ".join(L12_ORDER)
    text = f"""# Hist2Pheno benchmark input

Seven public training cohorts (CODEX ESCC is not included). Each folder is one dataset. `HEimgs/` holds the H&E that has nucleus ground truth. `metadata/` holds one CSV per image: every nucleus that maps onto the shared 3-level label space.

These are the same labeled sections Hist2Pheno trains on ({len(slides)} images, {int(slides['n_nuclei'].sum()):,} nuclei). A nucleus whose native label is outside this ontology is not in the CSV. `slides.csv` column `n_raw_in_h5ad` is the count before that filter.

## Layout

```
Input_data/
  HEimgs/<dataset>/<sample>.tif
  metadata/<dataset>/<sample>.csv
  slides.csv
  ontology_labels.csv
  label_counts.csv
```

The TIFF in `HEimgs` is a hard link to the file in this repository, not a second copy. Copying this folder onto another disk (rsync, cp, tar) writes the real image bytes.

## Nucleus table

| column | meaning |
|---|---|
| `cell_id` | nucleus id inside that image |
| `x_px`, `y_px` | centroid in the paired TIFF. Origin is the top-left corner. `x_px` increases to the right, `y_px` increases downward. Column = `x_px`, row = `y_px`. |
| `level1` | lineage, 5 classes |
| `level12` | sub-lineage, 8 classes |
| `level2` | fine cell type, 20 classes |
| `native_label` | the cohort's own name before it was mapped onto the three levels |

`level2` determines `level12` and `level1`. The parent of each `level2` is `ontology_labels.csv`.

**level1 (5):** {l1}

**level12 (8):** {l12}

## Pixel size

`x_px` / `y_px` are pixels of the file in `HEimgs`, before any rescale. `slides.csv` column `um_per_px` is the physical size Hist2Pheno used for that cohort. Hist2Pheno itself crops a patch after scaling the image to 0.5 µm/px. GBM is the exception: those two microscope TIFFs were used at their native pixel grid and were not rescaled, so `um_per_px` is blank.

| dataset | tech | images | nuclei | µm/px |
|---|---|---:|---:|---:|
{table}

## What this is not

StarDist segmentations of unlabeled nuclei are not included. Only nuclei with a ground-truth label on this 3-level ontology are in the CSV.
"""
    (INPUT_ROOT / "README.md").write_text(text)


def main() -> None:
    index = pd.read_csv(INDEX_CSV)
    index = index.loc[index["ok"].astype(str).str.lower().isin(["true", "1"])].copy()
    ontology = load_pancancer_ontology()
    onto_names = set(ontology["L2"].astype(str))
    if len(onto_names) != 20:
        raise RuntimeError(f"expected 20 L2 classes, got {len(onto_names)}")

    records = []
    count_rows = []
    n = len(index)
    for i, (_, row) in enumerate(index.iterrows(), 1):
        rec = export_section(row, onto_names)
        count_rows.extend(rec.pop("_counts"))
        if rec["n_nuclei"] != rec["n_keep_in_index"]:
            print(
                f"  WARN {rec['dataset']}/{rec['sample']}: "
                f"exported {rec['n_nuclei']} vs index {rec['n_keep_in_index']}",
                flush=True,
            )
        records.append(rec)
        if i % 50 == 0 or i == n:
            print(f"  {i}/{n}  {rec['dataset']}/{rec['sample']}", flush=True)

    slides = pd.DataFrame(records)
    order = list(FOLDER.values())
    slides["dataset"] = pd.Categorical(slides["dataset"], order, ordered=True)
    slides = slides.sort_values(["dataset", "sample"]).reset_index(drop=True)
    slides.to_csv(INPUT_ROOT / "slides.csv", index=False)

    labels = ontology.rename(columns={"L1": "level1", "L12": "level12", "L2": "level2"})
    keep_cols = [c for c in ("level2", "level12", "level1") if c in labels.columns]
    labels[keep_cols].to_csv(INPUT_ROOT / "ontology_labels.csv", index=False)

    pd.DataFrame(count_rows).to_csv(INPUT_ROOT / "label_counts.csv", index=False)
    write_readme(slides, labels[keep_cols])

    print("\nslides", len(slides), "nuclei", int(slides["n_nuclei"].sum()))
    print(slides.groupby("dataset", observed=True)["n_nuclei"].agg(["size", "sum"]).to_string())


if __name__ == "__main__":
    main()
