"""Spot-level validation of Hist2Pheno nuclei predictions against Visium.

2026.11.16. Three independent references, all on the same Visium spots:

* cell2location abundances (NPC): compared as **proportions** (row-normalised
  per spot) and as **counts** (Hist2Pheno expected count = sum of softmax of
  the nuclei in the spot), never proportion against count.
* marker-gene scores from ``filtered_feature_bc_matrix.h5`` (any Visium
  cohort): the Visium RNA itself is the ground truth for "is there B / T /
  myeloid / stroma / epithelium in this spot".
* pathologist spot annotations (Prostate, Erickson et al. 2022
  ``*_Final_Consensus_Annotations.csv``): per-annotation composition and
  one-vs-rest AUROC.

Spots are matched to nuclei with ``nuclei_to_spots``; the capture disc is
55 µm, i.e. ``spot_diameter_fullres * 55 / 65`` (Space Ranger defines
``spot_diameter_fullres`` for a 65 µm spot).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from nuclei_to_spots import nuclei_to_spots

CAPTURE_UM = 55.0
SPACERANGER_SPOT_UM = 65.0

# L2 -> broad compartment. DCIS is neoplastic, so it is tumour here.
L2_TO_COMPARTMENT = {
    "Tumor": "Tumor", "DCIS": "Tumor",
    "Epithelial": "Epithelial", "Alveolar": "Epithelial",
    "Injury_epithelial": "Epithelial", "Myoepithelial": "Epithelial",
    "Fibroblast": "Stroma", "Myofibroblast": "Stroma", "Stromal": "Stroma",
    "Endothelial": "Endothelial",
    "CD4_T": "T", "CD8_T": "T", "NK": "T", "Tregs": "T",
    "B_cell": "B", "Plasma": "Plasma",
    "Macrophage": "Myeloid", "DC": "Myeloid", "Neutrophil": "Myeloid",
    "Neural": "Neural",
}
COMPARTMENTS = ("Tumor", "Epithelial", "Stroma", "Endothelial", "T", "B", "Plasma", "Myeloid", "Neural")

# Marker genes, human. Kept short and specific; Plasma uses MZB1/JCHAIN
# rather than immunoglobulin genes, which diffuse across spots.
MARKERS = {
    "B": ("MS4A1", "CD79A", "CD79B", "CD19", "BANK1"),
    "Plasma": ("MZB1", "JCHAIN", "DERL3", "TNFRSF17"),
    "T": ("CD3D", "CD3E", "CD2", "TRAC", "CD247"),
    "Myeloid": ("CD68", "CD14", "LYZ", "C1QA", "C1QB", "CD163"),
    "Stroma": ("COL1A1", "COL1A2", "COL3A1", "DCN", "LUM"),
    "Endothelial": ("PECAM1", "VWF", "CDH5", "CLDN5"),
    "Epithelial": ("EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"),
}
# Cohort-specific tumour markers (optional, pass to marker_scores).
TUMOR_MARKERS = {
    "Prostate": ("AMACR", "ERG", "TMEFF2", "PCA3", "FOLH1"),
    "NPC": ("KRT5", "KRT17", "S100A2", "SFN"),
}


def capture_radius_px(scalefactors_json) -> float:
    """Radius in full-res pixels of the 55 µm Visium capture area."""
    sf = json.loads(Path(scalefactors_json).read_text())
    return float(sf["spot_diameter_fullres"]) * CAPTURE_UM / SPACERANGER_SPOT_UM / 2.0


def read_tissue_positions(visium_dir) -> pd.DataFrame:
    """In-tissue spots: barcode, x (= pxl_col), y (= pxl_row) in full-res pixels.

    Handles Space Ranger 1.x ``tissue_positions_list.csv`` (no header) and 2.x
    ``tissue_positions.csv`` (header).
    """
    d = Path(visium_dir)
    for name in ("tissue_positions.csv", "tissue_positions_list.csv"):
        p = d / name
        if p.is_file():
            break
    else:
        raise FileNotFoundError(f"no tissue_positions*.csv under {d}")
    head = pd.read_csv(p, nrows=1, header=None).iloc[0].astype(str).tolist()
    if head[0] == "barcode":
        df = pd.read_csv(p)
    else:
        df = pd.read_csv(p, header=None, names=["barcode", "in_tissue", "array_row", "array_col",
                                               "pxl_row_in_fullres", "pxl_col_in_fullres"])
    df = df.loc[df["in_tissue"].astype(int) == 1].reset_index(drop=True)
    return pd.DataFrame({
        "barcode": df["barcode"].astype(str).to_numpy(),
        "x": df["pxl_col_in_fullres"].to_numpy(np.float64),
        "y": df["pxl_row_in_fullres"].to_numpy(np.float64),
    })


def group_columns(mean_probs, class_names, mapping: Mapping[str, str] = L2_TO_COMPARTMENT,
                  groups: Sequence[str] = COMPARTMENTS) -> pd.DataFrame:
    """Sum L2 columns into groups (unmapped L2 classes are dropped)."""
    P = np.asarray(mean_probs, dtype=np.float64)
    names = [str(c) for c in class_names]
    out = {}
    for g in groups:
        cols = [i for i, n in enumerate(names) if mapping.get(n) == g]
        out[g] = P[:, cols].sum(1) if cols else np.zeros(len(P))
    return pd.DataFrame(out)


def spot_h2p(nuclei_xy, probs, class_names, spots: pd.DataFrame, radius_px: float,
             mapping: Mapping[str, str] = L2_TO_COMPARTMENT,
             groups: Sequence[str] = COMPARTMENTS) -> pd.DataFrame:
    """Per spot: n_nuclei, mean softmax per group (``frac_*``) and expected
    counts (``count_*`` = sum of softmax). Spots without nuclei get NaN fractions."""
    cap = nuclei_to_spots(np.asarray(nuclei_xy, float), spots[["x", "y"]].to_numpy(float),
                          float(radius_px), shape="circle", values=np.asarray(probs, float))
    n = np.asarray(cap["n_nuclei"], dtype=np.float64)
    frac = group_columns(cap["mean"], class_names, mapping, groups)
    frac[n == 0] = np.nan
    out = pd.DataFrame({"barcode": spots["barcode"].to_numpy(), "n_nuclei": n})
    for g in groups:
        out[f"frac_{g}"] = frac[g].to_numpy()
        out[f"count_{g}"] = np.nan_to_num(frac[g].to_numpy()) * n
    return out


def proportions(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise non-negative abundances (rows summing to 0 become NaN)."""
    a = counts.to_numpy(np.float64)
    s = a.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(s > 0, a / s, np.nan)
    return pd.DataFrame(p, index=counts.index, columns=counts.columns)


def cell_fractions(counts: pd.DataFrame) -> pd.Series:
    """Slide-level cell proportion from per-spot abundances (sum, not argmax)."""
    tot = counts.clip(lower=0).sum(0)
    return tot / tot.sum()


def read_10x_h5_genes(h5_path, genes: Sequence[str], barcodes: Sequence[str] | None = None) -> pd.DataFrame:
    """Raw UMI counts for ``genes`` (spots × genes) from a Space Ranger v3 h5.

    Missing genes come back as all-zero columns. ``barcodes`` reorders rows
    (barcodes absent from the matrix become 0).
    """
    import h5py
    from scipy.sparse import csc_matrix

    genes = list(dict.fromkeys(str(g) for g in genes))
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in m["features"]["name"][:]]
        bcs = [x.decode() if isinstance(x, bytes) else str(x) for x in m["barcodes"][:]]
        shape = tuple(int(v) for v in m["shape"][:])           # genes × barcodes
        X = csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        total = np.asarray(X.sum(0)).ravel()
    pos = {}
    for i, n in enumerate(names):
        pos.setdefault(n, i)
    rows = [pos.get(g, -1) for g in genes]
    Xr = X.tocsr()
    data = np.zeros((len(bcs), len(genes)), dtype=np.float64)
    for j, r in enumerate(rows):
        if r >= 0:
            data[:, j] = Xr[r].toarray().ravel()
    df = pd.DataFrame(data, index=bcs, columns=genes)
    df["__total_umi"] = total
    if barcodes is not None:
        df = df.reindex([str(b) for b in barcodes]).fillna(0.0)
    return df


def marker_scores(h5_path, barcodes, marker_sets: Mapping[str, Sequence[str]] = MARKERS,
                  target_sum: float = 1e4) -> pd.DataFrame:
    """Per spot, mean log1p(CP10k) of each marker set's genes that exist in the matrix."""
    all_genes = sorted({g for gs in marker_sets.values() for g in gs})
    raw = read_10x_h5_genes(h5_path, all_genes, barcodes)
    tot = raw.pop("__total_umi").to_numpy(np.float64)
    norm = np.log1p(raw.to_numpy(np.float64) / np.maximum(tot, 1.0)[:, None] * target_sum)
    norm = pd.DataFrame(norm, index=raw.index, columns=raw.columns)
    present = raw.columns[(raw.to_numpy() > 0).any(0)]
    out = {}
    for name, gs in marker_sets.items():
        use = [g for g in gs if g in present]
        out[name] = norm[use].mean(axis=1) if use else pd.Series(np.nan, index=norm.index)
    res = pd.DataFrame(out, index=raw.index)
    res.attrs["genes_used"] = {k: [g for g in v if g in present] for k, v in marker_sets.items()}
    return res


def agreement(h2p: pd.DataFrame, ref: pd.DataFrame, pairs: Mapping[str, str] | None = None,
              *, min_spots: int = 10) -> pd.DataFrame:
    """Spearman / Pearson between matched columns over spots finite in both."""
    from scipy.stats import pearsonr, spearmanr

    pairs = pairs or {c: c for c in ref.columns if c in h2p.columns}
    rows = []
    for a, b in pairs.items():
        x = np.asarray(h2p[a], float)
        y = np.asarray(ref[b], float)
        ok = np.isfinite(x) & np.isfinite(y)
        n = int(ok.sum())
        if n < min_spots or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            rows.append({"h2p": a, "reference": b, "n_spots": n, "spearman": np.nan,
                         "spearman_p": np.nan, "pearson": np.nan})
            continue
        r, p = spearmanr(x[ok], y[ok])
        rows.append({"h2p": a, "reference": b, "n_spots": n, "spearman": float(r),
                     "spearman_p": float(p), "pearson": float(pearsonr(x[ok], y[ok])[0])})
    return pd.DataFrame(rows)


# Erickson consensus labels -> broad region. Matched case-insensitively.
_ANNOTATION_RULES = (
    (r"\bgg\s*\d|gleason|cancer|tumou?r|cribriform|carcinoma", "Tumor"),
    (r"\bpin\b|transition", "Precursor"),
    (r"benign|normal|gland|epithel", "Benign_epithelium"),
    (r"inflam|immune|lymph|tls", "Immune"),
    (r"strom|fibro|muscle|smooth", "Stroma"),
    (r"nerve|neur|ganglion", "Nerve"),
    (r"vessel|vascul|endoth", "Vessel"),
)


def annotation_region(label: str) -> str:
    s = str(label).strip().lower()
    if not s or s in ("nan", "exclude", "excluded", "na"):
        return "Exclude"
    for pat, region in _ANNOTATION_RULES:
        if re.search(pat, s):
            return region
    return "Other"


def read_annotations(path) -> pd.Series:
    """Barcode -> consensus label from ``*_Final_Consensus_Annotations.csv``."""
    df = pd.read_csv(path)
    bc_col = next((c for c in df.columns if str(c).lower() in ("barcode", "barcodes")), df.columns[0])
    lab_col = next((c for c in df.columns if c != bc_col), None)
    if lab_col is None:
        raise ValueError(f"{path} has no label column")
    return pd.Series(df[lab_col].astype(str).to_numpy(), index=df[bc_col].astype(str).to_numpy(),
                     name="annotation")


# Which Hist2Pheno group should be high in which annotated region.
REGION_EXPECT = {
    "Tumor": ("Tumor",),
    "Benign_epithelium": ("Epithelial",),
    "Immune": ("T", "B", "Plasma", "Myeloid"),
    "Stroma": ("Stroma",),
    "Vessel": ("Endothelial",),
}


def _auroc(score, positive) -> float:
    from scipy.stats import rankdata

    score = np.asarray(score, float)
    pos = np.asarray(positive, bool)
    ok = np.isfinite(score)
    score, pos = score[ok], pos[ok]
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def annotation_report(h2p: pd.DataFrame, regions: pd.Series,
                      groups: Sequence[str] = COMPARTMENTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(mean Hist2Pheno fraction per region, one-vs-rest AUROC per expected group)."""
    df = h2p.copy()
    df["region"] = regions.reindex(df["barcode"]).to_numpy()
    df = df.loc[df["region"].notna() & (df["region"] != "Exclude") & (df["n_nuclei"] > 0)]
    cols = [f"frac_{g}" for g in groups]
    mean = df.groupby("region")[cols].mean()
    mean.columns = list(groups)
    mean.insert(0, "n_spots", df.groupby("region").size())
    rows = []
    for region, gs in REGION_EXPECT.items():
        if not (df["region"] == region).any():
            continue
        score = df[[f"frac_{g}" for g in gs]].sum(axis=1)
        rows.append({"region": region, "h2p_groups": "+".join(gs),
                     "n_pos": int((df["region"] == region).sum()), "n_neg": int((df["region"] != region).sum()),
                     "auroc": _auroc(score, df["region"] == region)})
    return mean, pd.DataFrame(rows)
