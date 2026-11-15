# Hist2Pheno package

Shared, dataset-neutral modeling, transformation, and plotting code for Hist2Pheno.
Dataset directories such as `CODEX_hcc/` and `Xenium_lung/` remain responsible
for data paths, file I/O, cohort-specific orchestration, and report assembly.

## 2026-11-16 A8: class-weighted L12 (opt-in)

A0–A7 leave L12 unweighted. L1 and L2 cross-entropy are `balanced`, but `CE(L12)` and the two aggregated NLL terms (`L2→L12`, `L12→L1`) were not. That unweighted L12 term pulls rare families, including `B_Plasma` (about 3.5% of cells), back toward the empirical prior. A7 tried to fix the same hole with a hierarchical sampler and focal loss, and the pooled OOF L2 fell to the A0 level. A8 does not repeat that. It keeps the A3 recipe and only class-weights the L12 terms.

`--l12-class-weight-mode` is `none` or `balanced`. `none` is the default.

| | `CE(L12)` | `NLL(L2→L12)` | `NLL(L12→L1)` | two JS terms |
|--|-----------|---------------|---------------|--------------|
| `none` (A0–A7) | unweighted | unweighted | unweighted | unweighted |
| `balanced` (A8) | inverse-frequency on the 8 L12 classes | same L12 weights | the existing L1 `balanced` weights | still unweighted |

`weight=None` is bit-identical to the previous `adjacent_loss`. The extra key is written into the training protocol and `loss_weights` only when the mode is not `none`, so finished A0–A7 checkpoint families stay the same and their fold checkpoints remain reusable. A8 is the A3 trunk (k=8 mean, learned gate, same scalar loss weights) plus `--l12-class-weight-mode balanced`. The command and the output folder are in [`code/PanCancer/README.md`](../PanCancer/README.md#2026-11-16-a8-balanced-l12-not-yet-trained).

Check: `python -m pytest -q code/Hist2Pheno_pkg/test_l12_class_weight.py`.

## 2026-11-15 Integrity fixes

These fixes do not change trained weights or the published OOF AUROC of a finished run. They close paths where a score could be attached to the wrong class, the wrong parent, or a file from another run. Archives written before this date keep their old contents until that step is run again. The Pan-cancer OOF file contract is in [`code/PanCancer/README.md`](../PanCancer/README.md#2026-11-15-oof-archive-contract).

### L2→L1 aggregation at evaluation (`model.py`)

Fold metrics used to infer each L2 class's L1 parent by majority vote on the labels present in that evaluation array. An L2 class with no cells kept parent −1, so its softmax mass never reached an L1 bin. Evaluation now reads `hierarchy_maps` from the checkpoint (`L2 → L12 → L1`), including classes absent from the subset, and takes the L1 width from the loaded head. On the current workbook that composition equals the direct L2→L1 map (20 / 8 / 5), which is the map training already uses to fill missing parents. `_load_fold_model_for_eval` attaches `checkpoint_metadata` and `checkpoint_scaler` so the map is available. L2 head AUROC does not use this aggregation.

### Confidence caches and class order (`prediction_confidence.py`)

`conformal_from_oof` and `conformal_hierarchy` read `l2_class_names`, `l12_class_names`, and `l1_head_class_names` from `oof_predictions.npz` when those arrays exist, and match columns by name. A test file whose column order differs from the OOF file is not assumed to be the same order. An older OOF file without those arrays still falls back to the test-file order. That fallback is safe only when both files come from the same training run.

Confidence sidecars are keyed by a hash of the input file bytes, the options, and this module. A changed probability file or a changed option does not reuse the old sidecar. The write goes to a temporary `.npz` and then replaces the destination. The cluster sidecar also stores `obs_names` and `class_names`. Sidecars from before this date have no `input_signature` and are recomputed on the next call.

### External probability archives (`external_consistency.py`)

`{sample}_l2_probs.npz` now also stores `checkpoint_family`, `l1_probs`, `l1_class_names`, `l12_probs`, and `l12_class_names`, in the checkpoint head order. The prediction stamp includes this module, so the next predict does not treat an older L2-only archive as complete. `checkpoint_family` is how a later check refuses to compare an A2 archive with an A3 OOF file.

### Repeated label remap (`pancancer_labels.py`)

`remap_obs_frame` writes canonical names into `final_CT` and keeps the original string in `native_CT`. Some canonical names are not valid native keys (`B_cell`, `Tumor`, `DC`, `Neural` do not map again). A second call now remaps `native_CT` when that column is already present. Checked on a real native: `Invasive_Tumor` stays `Tumor` on the second call. The Pan-cancer training loader still maps the source column on each read; it does not depend on this second-call guard.

### Spatial graph reuse (`spatial_context.py`)

`prepare_runtime` used to keep `_context_graphs` for the life of the CV object. The cache key is now the spatial config, the coordinates, and the section ids. A new `k`, radius, or coordinate array rebuilds the neighbor graph.

### Read-only B / Plasma audit (`audit_uni2_bcell.py`)

This script does not change labels, probabilities, or weights. It checks that an external probability archive and an OOF archive belong to the selected checkpoint's family and L2 class order, then reports softmax mass — not a cell count — for `B_cell`, `Plasma`, and `B_Plasma`. When the archive contains L12 probabilities, it also compares the L12 head with L2 mass rolled up by `hierarchy_maps`.

```bash
python code/Hist2Pheno_pkg/audit_uni2_bcell.py \
  --checkpoint path/to/best_mlp_gpu.pt \
  --pred-probs path/to/sample_l2_probs.npz \
  --oof path/to/oof_predictions.npz \
  --oof-rows path/to/oof_rows.csv \
  --output path/to/audit_out
```

`fold_ids` inside `oof_predictions.npz` follow the probability rows (fold-concatenated order). `oof_rows.csv` stays in global row order. Move between them with `row_indices`.

## 2026-11-11 Per-nucleus cell-type confidence

[`prediction_confidence.py`](prediction_confidence.py) scores the L2 softmax
that `external_validation.cmd_predict` already writes
(`{sample}_l2_probs.npz`). It does **not** retrain the MLP. Spatial maps
currently `argmax` that vector. Morphologically similar types (B vs Plasma vs
T, CD4 vs CD8) keep a low peak, so they vanish from the winner map even when
the mass on the parent class (L12 `B_Plasma` / `T_cell`) is real. The module
adds a per-nucleus, per-type confidence instead of depending only on a larger
encoder.

Two layers, both model-agnostic:

1. **Softmax diagnostics** (no labels). Max probability, margin to the
   runner-up, normalized entropy, and a collapse to a coarser label when the
   top two classes sit in one similarity family (`B_cell`+`Plasma` →
   `B_Plasma`; CD4/CD8/Treg → `T_cell`). Weak peaks become `low_confidence`.
2. **Conformal presence** in the sense of Jin et al., bioRxiv
   2026.03.01.708850
   ([UTOPIA](https://github.com/kaitianjin/UTOPIA)). Pan-cancer OOF cells
   (`oof_predictions.npz`) are the calibration set. For each class `c` the
   null is “`c` is absent”. Nuclei whose `P(c)` exceeds the OOF false-positive
   scores get a small conformal p-value. **Confidence** is the UTOPIA
   log-stretch of that *raw* p. **Presence** is a calibrated FPR gate:
   `P(c)` above the `(1-α)` quantile of OOF cells that are not class `c`
   (`--fdr 0.10` → 90th percentile of the null). Whole-slide BH on ~1e5
   nuclei is stored as `p_adj` but is not the map threshold: UTOPIA BH is
   per histology cluster, and OOF softmax false positives already reach ~1,
   so `p_adj <= 0.10` is almost always empty on an NPC WSI.

External slides such as NPC have no nucleus GT. Calibration must come from
the training OOF, not from the test slide (UTOPIA’s out-of-subject setting).
Coarser targets (`B_Plasma`, `lymphoid`) are usually more confident than
B vs T at 7–8 µm, which is the same granularity result in that paper.

This is output-side quantification. If almost every nucleus has `P(B)` below
the OOF-null 90th percentile, the FPR gate stays empty — that is the model
saying it is not sure, not a cell2location-style abundance correction.
Training-side balance (focal loss, hierarchical sampler) is a separate lever.

Add `code/Hist2Pheno_pkg` to `PYTHONPATH`. Use the SeededNTM interpreter.

```python
from pathlib import Path
from prediction_confidence import (
    annotate_nuclei,
    attach_to_obs,
    conformal_from_oof,
    load_l2_probs,
)

names, probs = load_l2_probs(pred_dir / f"{sample}_l2_probs.npz")
table = annotate_nuclei(probs, names, margin_min=0.08, max_prob_min=0.25)
# table["label_plot"]: sharp L2, parent family, or "low_confidence"

conf = conformal_from_oof(
    probs,
    names,
    oof_npz=Path("data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/oof_predictions.npz"),
    classes=["B_cell", "Plasma"],
    groups={
        "B_Plasma": ("B_cell", "Plasma"),
        "lymphoid": ("B_cell", "Plasma", "CD4_T", "CD8_T", "Tregs", "NK"),
    },
)
# presence at 10% FPR vs OOF-null: conf["B_cell"]["present"]
# display score (raw conformal p):  conf["B_cell"]["confidence"]
obs = attach_to_obs(adata.obs, table, conformal=conf, fdr=0.10)
```

CLI (writes a sidecar `{sample}_confidence.npz`, does not rewrite the pred h5ad):

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
/nobackup2/users/lingyu/conda_envs/SeededNTM/bin/python \
  code/Hist2Pheno_pkg/prediction_confidence.py \
  data/Visium/NPC/pred_uni2_A2_bestmlp_paired_v2_seed42/<sample>_l2_probs.npz \
  --oof data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/oof_predictions.npz \
  --classes B_cell Plasma B_Plasma \
  --fdr 0.10
```

`--classes` accepts L2 names and family names from `DEFAULT_SIMILAR_FAMILIES`
(`B_Plasma`, `T_cell`, `lymphoid`, `myeloid`, `epithelial`). Tests live in
[`test_prediction_confidence.py`](test_prediction_confidence.py).

## 2026-20-25 Which embedder to run next

`data/` currently keeps **UNI** results only. The paired UNI2 run is `--method UNI2` with `--patch_size 14` (one class token, 7 µm at 0.5 µm/px), the same one-token crop UNI uses at 16 px / 8 µm. Commands for the seven training cohorts are in the next section. `--patch_size 28` (14 µm, four tokens, still one class token) is the larger single-cell crop if a later run should include cytoplasm. Write `.pth` files to a new directory and retrain the MLP. The published pan-cancer checkpoint stays the UNI one.

Hist2Pheno labels one nucleus at a time. The H&E is scaled to 0.5 µm/px, a square is cropped on the centroid, and one vector goes into the multi-head MLP (L2 / L12 / L1). Neighbor context is a later residual on those vectors (`SpatialContextFusion` in `base.py`). TLS / SRI / TNI are computed from the predicted labels and coordinates. The embedder’s job is the morphology of that cell. A crop that already mixes the surrounding cells makes two adjacent nuclei with different ground-truth labels look alike, and the L2 head (CD4 / CD8 / Treg, macrophage / DC, fibroblast / myofibroblast) is the head that suffers.

What each `--method` actually saves, given `Image_feature_extraction.py`:

| `--method` | Vector the `.pth` stores | Crop that matches this code | Field at 0.5 µm/px | Fit to a per-nucleus label |
|------------|--------------------------|-----------------------------|--------------------|----------------------------|
| `UNI` | class token, 1024-d | 16 px = one ViT-L/16 token | 8 µm | current results; native token equals the nucleus crop |
| **`UNI2`** | class token, 1536-d | **28 px = 2×2 ViT-H/14 tokens** | **14 µm** | same one-vector contract; crop still covers one cell |
| `H-optimus-1` | class token, 1536-d | 224 px required | 112 µm | one vector per centroid, but the tile contains many cells |
| `HIPT` | patch tokens after dropping token 0, concatenated | 16 px → a single 384-d token | 8 µm | bottom ViT-256 stage only; the 4K hierarchy is unused |
| `Virchow2` | every token after the first five, concatenated | 16 px is not a multiple of 14 | — | a bag of patch tokens, flattened into the MLP |

**UNI2-h** is the one that keeps the UNI contract and upgrades the encoder.

- The loader already stores one class token and flattens it. `input_dim` is read from that vector, so 1024 → 1536 needs a new training run and no change to matching, h5ad I/O, or the spatial residual. UNI2-h is the later MahmoodLab encoder (ViT-H/14, 1536-d) trained as the successor to UNI; the model card reports stronger H&E tile features than UNI. That is the signal this MLP reads.
- Its token is 14 px, so the crop has to be a multiple of 14. `patch_size=28` is 14 µm: nucleus plus a rim of cytoplasm, pooled by the class token over four tokens. That is the smallest crop on which the class token does the pooling UNI2 was trained to do, and it stays on one cell. `patch_size=14` (7 µm, one token) also runs, and is the closer twin of the UNI 8 µm crop if the comparison must hold the field fixed. `patch_size=16` only triggers a warning: 16 is not an integer number of 14 px tokens, so that crop is not a UNI2 input.
- Weights are gated (~2.6 GiB). Accept the license, then follow [`code/UNI2/assets/ckpts/uni2-h/README.md`](../UNI2/assets/ckpts/uni2-h/README.md).

The other three answer a different question than “replace UNI on the cells already embedded.”

- **H-optimus-1** is a ViT-G/14 trained on 224×224 tiles at 0.5 µm/px, and this script refuses any other `patch_size`. At that size the window is 112 µm, so centroids tens of microns apart share almost the same pixels. The embedding is a tissue tile. It is the right scale for a later context-window ablation, and the wrong scale for the first non-UNI cell run. The weights are **CC BY-NC-ND 4.0** (~4.3 GiB), which also constrains redistribution of anything derived from them.
- **Virchow2** is a strong tile encoder, but this script drops the first five tokens and saves the rest. The h5ad loader concatenates them, so a 224 px tile becomes hundreds of 1280-d tokens in one MLP input. A 16 px nucleus crop is not a multiple of its 14 px token. Using it as a UNI replacement means changing the representation, not only the weights.
- **HIPT** at `patch_size=16` keeps one 384-d token from `vit256_small_dino`. That model was trained on 256×256 tiles, and the second stage that makes HIPT hierarchical is not called. The vector is smaller and comes from an earlier pretraining recipe than UNI.

Practical split from the UNI tree: new folder for UNI2 `.pth`, new h5ad, new MLP checkpoints. Leave the UNI embeddings in place so the published track and this run stay comparable.

## UNI2 `patch_size=14` on the seven training cohorts

Same slides, same coordinate CSVs, and the same `scale` / `scale_image` as the UNI extractors. The only changes are `--method UNI2`, `--patch_size 14`, and the output directory. Each `.pth` is one 1536-d class token.

Outputs sit beside the UNI tree and do not replace it:

| Cohort | UNI directory this mirrors | UNI2 directory |
|--------|----------------------------|----------------|
| Xenium lung, 25 Complete | `{sample}_project_all_UNI/ImgEmbeddings_all/sc_pth_16_16` | `{sample}_project_all_UNI2/.../sc_pth_14_14` |
| Xenium BRCA, rep1 / rep2 | `Cases/{sample}/project_all_UNI/.../sc_pth_16_16` | `project_all_UNI2/.../sc_pth_14_14` |
| Xenium CRC, P1 / P2 / P5 | same pattern | same pattern |
| CODEX HCC, 36 annotated | `HE/{HE_KEY}/project_all_UNI/.../sc_pth_16_16` | `project_all_UNI2/.../sc_pth_14_14` |
| CODEX PDAC, 278 annotated | `s1167/{ACQ}/project_all_UNI/.../sc_pth_16_16` | `project_all_UNI2/.../sc_pth_14_14` |
| CODEX GIST, 550 annotated | same pattern | same pattern |
| CODEX GBM, Initial / Recurrent | `Cases/{sample}/project_all_UNI/.../sc_pth_16_16` | `project_all_UNI2/.../sc_pth_14_14` |

`gt` is the training coordinate file. `stardist` is the same crop on every segmented nucleus, which the UNI track extracts next. Lung Incomplete, PDAC Incomplete, and CODEX ESCC are not in this script. Accept the UNI2-h license first ([`code/UNI2/assets/ckpts/uni2-h/README.md`](../UNI2/assets/ckpts/uni2-h/README.md)). The third argument is the physical GPU index. The script activates the **SeededNTM** conda env, then sets `CUDA_VISIBLE_DEVICES` to that index, so the extractor sees it as `cuda:0`. A sample whose `sc_pth_14_14/` already contains `.pth` files is skipped, so a stopped run can be started again.

The runnable file is [`run_uni2_patch14.sh`](run_uni2_patch14.sh). The same two blocks are comments at the top of that file. Run from the repo root, one cohort at a time. `CUDA_DEVICE` is the physical GPU index.

`gt` writes `ImgEmbeddings_all/sc_pth_14_14` from the matched training coordinates:

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
CUDA_DEVICE=0
bash code/Hist2Pheno_pkg/run_uni2_patch14.sh lung gt "${CUDA_DEVICE}"
bash code/Hist2Pheno_pkg/run_uni2_patch14.sh brca gt "${CUDA_DEVICE}"
bash code/Hist2Pheno_pkg/run_uni2_patch14.sh crc gt "${CUDA_DEVICE}"
bash code/Hist2Pheno_pkg/run_uni2_patch14.sh hcc gt "${CUDA_DEVICE}"
bash code/Hist2Pheno_pkg/run_uni2_patch14.sh pdac gt "${CUDA_DEVICE}"
bash code/Hist2Pheno_pkg/run_uni2_patch14.sh gist gt "${CUDA_DEVICE}"
bash code/Hist2Pheno_pkg/run_uni2_patch14.sh gbm gt "${CUDA_DEVICE}"
```

`stardist` writes `ImgEmbeddings_all_stardist/sc_pth_14_14`, one token per segmented nucleus. `SAVE_PATCH_IMAGES=0` skips the PNG crops. Those crops are large on the all-nuclei sets; CRC P1 and P5 already skip them when the variable is unset, matching the UNI run. `SAVE_PATCH_IMAGES=1` keeps them. One sample: `SAMPLE=VUHD113` (lung), `SAMPLE=rep1` (BRCA), `SAMPLE=P2CRC` (CRC), `SAMPLE=P174511_Initial` (GBM), `ACQ_ID=...` (PDAC / GIST), `HE_KEY=...` (HCC).

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
CUDA_DEVICE=0
SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh lung stardist "${CUDA_DEVICE}"
SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh brca stardist "${CUDA_DEVICE}"
SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh crc stardist "${CUDA_DEVICE}"
SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh hcc stardist "${CUDA_DEVICE}"
SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh pdac stardist "${CUDA_DEVICE}"
SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh gist stardist "${CUDA_DEVICE}"
SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh gbm stardist "${CUDA_DEVICE}"
```

BRCA all-nuclei prediction with the pan-cancer UNI2 A3 checkpoint needs the `brca stardist` line. The GT tokens in `ImgEmbeddings_all/sc_pth_14_14` are a different coordinate set and are not a substitute. After that extract, `code/Xenium_brca/BRCA_pred_uni2_pretrained.ipynb` scores every StarDist nucleus.

```bash
#!/usr/bin/env bash
# UNI2-h, patch_size=14, one class token per nucleus.
# Mirrors the UNI extractors. Writes project_all_UNI2 / sc_pth_14_14.
# Does not read or delete project_all_UNI.
#
# Run from the repo root. The last argument is the physical GPU. SeededNTM is activated inside.
# A cohort whose sc_pth_14_14/ already has .pth files is skipped.
#
# gt — matched training coordinates → ImgEmbeddings_all/sc_pth_14_14
# CUDA_DEVICE is the physical GPU index. The script maps it to cuda:0.
#   cd /home/lingyu/ssd2/Python/Hist2Pheno
#   CUDA_DEVICE=0
#   bash code/Hist2Pheno_pkg/run_uni2_patch14.sh lung gt "${CUDA_DEVICE}"
#   bash code/Hist2Pheno_pkg/run_uni2_patch14.sh brca gt "${CUDA_DEVICE}"
#   bash code/Hist2Pheno_pkg/run_uni2_patch14.sh crc gt "${CUDA_DEVICE}"
#   bash code/Hist2Pheno_pkg/run_uni2_patch14.sh hcc gt "${CUDA_DEVICE}"
#   bash code/Hist2Pheno_pkg/run_uni2_patch14.sh pdac gt "${CUDA_DEVICE}"
#   bash code/Hist2Pheno_pkg/run_uni2_patch14.sh gist gt "${CUDA_DEVICE}"
#   bash code/Hist2Pheno_pkg/run_uni2_patch14.sh gbm gt "${CUDA_DEVICE}"
#
# stardist — every segmented nucleus → ImgEmbeddings_all_stardist/sc_pth_14_14
# SAVE_PATCH_IMAGES=0 skips the PNG crops.
#   cd /home/lingyu/ssd2/Python/Hist2Pheno
#   CUDA_DEVICE=0
#   SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh lung stardist "${CUDA_DEVICE}"
#   SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh brca stardist "${CUDA_DEVICE}"
#   SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh crc stardist "${CUDA_DEVICE}"
#   SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh hcc stardist "${CUDA_DEVICE}"
#   SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh pdac stardist "${CUDA_DEVICE}"
#   SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh gist stardist "${CUDA_DEVICE}"
#   SAVE_PATCH_IMAGES=0 bash code/Hist2Pheno_pkg/run_uni2_patch14.sh gbm stardist "${CUDA_DEVICE}"

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <lung|brca|crc|hcc|pdac|gist|gbm> <gt|stardist> <gpu>" >&2
  echo "  gpu is the physical index, e.g. 0 or 1. The script activates SeededNTM." >&2
  exit 1
fi
COHORT="$1"
COORD="$2"
GPU="$3"
if [[ ! "${COORD}" =~ ^(gt|stardist)$ ]]; then
  echo "ERROR: coord must be gt or stardist, got ${COORD}" >&2
  exit 1
fi
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: gpu must be a physical index such as 0 or 1, got ${GPU}" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU}"

CONDA_BASE="${CONDA_BASE:-/opt/anaconda3}"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: conda.sh not found: ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# SeededNTM's activate.d script reads unset binutils vars. nounset must be off here.
set +u
conda activate SeededNTM
set -u
hash -r
PYTHON="$(command -v python)"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found after conda activate SeededNTM" >&2
  exit 1
fi

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"

PATCH_SIZE=14
METHOD=UNI2
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
cd "${REPO}"
echo "conda env: SeededNTM"
echo "python:    ${PYTHON}"
echo "GPU:       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (visible inside Python as cuda:0)"

OK=0
SKIP=0
FAIL=0

run_one() {
  local dataset="$1" position="$2" he="$3" scale_image="$4" scale="$5" out_root="$6" save_png="$7"
  local pth_dir="${out_root}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}"
  if [[ ! -f "${he}" ]]; then
    echo "SKIP missing HE: ${he}"
    SKIP=$((SKIP + 1))
    return 0
  fi
  if [[ ! -f "${position}" ]]; then
    echo "SKIP missing position: ${position}"
    SKIP=$((SKIP + 1))
    return 0
  fi
  if [[ "${SKIP_IF_DONE}" == "1" && -d "${pth_dir}" ]]; then
    local n
    n="$(find "${pth_dir}" -maxdepth 1 -name '*.pth' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${n}" -gt 0 ]]; then
      echo "SKIP done (${n} .pth): ${pth_dir#${REPO}/}"
      SKIP=$((SKIP + 1))
      return 0
    fi
  fi
  mkdir -p "${out_root}"
  local -a args=(
    code/Image_feature_extraction.py
    --dataset "${dataset}"
    --position_path "${position}"
    --rawimage_path "${he}"
    --scale_image "${scale_image}"
    --method "${METHOD}"
    --patch_size "${PATCH_SIZE}"
    --output_img "${out_root}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}_image"
    --output_pth "${pth_dir}"
    --logging "${out_root}/"
  )
  if [[ "${scale_image}" == "True" ]]; then
    args+=(--scale "${scale}")
  fi
  if [[ "${save_png}" != "1" ]]; then
    args+=(--no_save_patch_images)
  else
    mkdir -p "${out_root}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}_image"
  fi
  echo "RUN ${dataset}"
  echo "  HE:       ${he#${REPO}/}"
  echo "  position: ${position#${REPO}/}"
  echo "  output:   ${pth_dir#${REPO}/}"
  if "${PYTHON}" "${args[@]}"; then
    OK=$((OK + 1))
  else
    echo "FAILED: ${dataset}" >&2
    FAIL=$((FAIL + 1))
  fi
}

case "${COHORT}" in
  lung)
    DATA="${REPO}/data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data"
    CASES="${DATA}/Complete_Cases"
    SCALE=0.425
    shopt -s nullglob
    for case_dir in "${CASES}"/*/; do
      sample="$(basename "${case_dir}")"
      if [[ -n "${SAMPLE:-}" && "${sample}" != "${SAMPLE}" ]]; then
        continue
      fi
      he_tifs=("${case_dir}"/*-HE.tif)
      if (( ${#he_tifs[@]} == 0 )); then
        echo "SKIP ${sample}: no *-HE.tif"
        SKIP=$((SKIP + 1))
        continue
      fi
      if [[ "${COORD}" == "gt" ]]; then
        pos="${case_dir}/${sample}_cells_partitioned_by_annotation_sample_match_with_pixel.csv"
        sub="ImgEmbeddings_all"
      else
        pos="${case_dir}/${sample}_Float_prob0.01_nms_0.3.csv"
        sub="ImgEmbeddings_all_stardist"
      fi
      run_one "${sample}" "${pos}" "${he_tifs[0]}" True "${SCALE}" \
        "${case_dir}/${sample}_project_all_UNI2/${sub}" "${SAVE_PATCH_IMAGES:-1}"
    done
    ;;
  brca)
    BRCA="${REPO}/data/Xemium/BRCA"
    SCALE="${SCALE:-0.84}"
    for sample in ${SAMPLE:-rep1 rep2}; do
      case "${sample}" in
        rep1) prefix="Xenium_FFPE_Human_Breast_Cancer_Rep1" ;;
        rep2) prefix="Xenium_FFPE_Human_Breast_Cancer_Rep2" ;;
        *) echo "ERROR: SAMPLE must be rep1 or rep2" >&2; exit 1 ;;
      esac
      he="${BRCA}/human_breast_Xenium_${sample}/${prefix}_he_image.tif"
      if [[ "${COORD}" == "gt" ]]; then
        pos="${BRCA}/Cases/${sample}/${sample}_cells_with_pixel.csv"
        sub="ImgEmbeddings_all"
      else
        key="${prefix}_he_image"
        pos="${BRCA}/StarDist_Segment/${key}/${key}_Float_prob0.01_nms_0.3.csv"
        sub="ImgEmbeddings_all_stardist"
      fi
      run_one "sc_${sample}" "${pos}" "${he}" True "${SCALE}" \
        "${BRCA}/Cases/${sample}/project_all_UNI2/${sub}" "${SAVE_PATCH_IMAGES:-1}"
    done
    ;;
  crc)
    CRC="${REPO}/data/Xemium/CRC"
    SCALE="${SCALE:-0.548}"
    for sample in ${SAMPLE:-P1CRC P2CRC P5CRC}; do
      case "${sample}" in
        P1CRC|P2CRC|P5CRC) ;;
        *) echo "ERROR: SAMPLE must be P1CRC, P2CRC, or P5CRC" >&2; exit 1 ;;
      esac
      folder="${sample/CRC/_CRC}"
      prefix="Xenium_V1_Human_Colon_Cancer_${folder}_Add_on_FFPE"
      he="${CRC}/HE_images/${prefix}_he_image.ome.tif"
      if [[ "${COORD}" == "gt" ]]; then
        pos="${CRC}/Cases/${sample}/${sample}_cells_with_pixel.csv"
        sub="ImgEmbeddings_all"
      else
        key="${prefix}_he_image.ome"
        pos="${CRC}/StarDist_Segment/${key}/${key}_Float_prob0.01_nms_0.3.csv"
        sub="ImgEmbeddings_all_stardist"
      fi
      save="${SAVE_PATCH_IMAGES:-}"
      if [[ -z "${save}" ]]; then
        if [[ "${sample}" == "P2CRC" ]]; then save=1; else save=0; fi
      fi
      run_one "sc_${sample}" "${pos}" "${he}" True "${SCALE}" \
        "${CRC}/Cases/${sample}/project_all_UNI2/${sub}" "${save}"
    done
    ;;
  hcc)
    S4769="${REPO}/data/CODEX/HCC/Michael_data_transfer/s4769"
    STARDIST_ROOT="${STARDIST_ROOT:-${REPO}/data/CODEX/HCC/StarDist_Segment}"
    export REPO
    mapfile -t ROWS < <("${PYTHON}" - <<'PY'
import sys
from pathlib import Path
repo = Path("/home/lingyu/ssd2/Python/Hist2Pheno")
sys.path.insert(0, str(repo / "code" / "CODEX_hcc"))
from match_codex_cells_with_pixel import list_aligned_annotated_regions
df = list_aligned_annotated_regions()
for _, row in df.iterrows():
    print(f"{row['CODEX_ACQUISITION_ID']}\t{row['MATCHED_HE']}")
PY
)
    if (( ${#ROWS[@]} == 0 )); then
      echo "ERROR: no annotated HCC regions" >&2
      exit 1
    fi
    for row in "${ROWS[@]}"; do
      acq="${row%%$'\t'*}"
      he_key="${row#*$'\t'}"
      if [[ -n "${ACQ_ID:-}" && "${acq}" != "${ACQ_ID}" ]]; then continue; fi
      if [[ -n "${HE_KEY:-}" && "${he_key}" != "${HE_KEY}" ]]; then continue; fi
      he="${S4769}/HE/${he_key}/figures/${he_key}.tif"
      if [[ "${COORD}" == "gt" ]]; then
        pos="${S4769}/${acq}/${acq}.cell_data.csv"
        dataset="sc_${acq}"
        sub="ImgEmbeddings_all"
      else
        pos="${STARDIST_ROOT}/${he_key}/${he_key}_Float_prob0.01_nms_0.3.csv"
        dataset="sc_${he_key}"
        sub="ImgEmbeddings_all_stardist"
      fi
      run_one "${dataset}" "${pos}" "${he}" False "" \
        "${S4769}/HE/${he_key}/project_all_UNI2/${sub}" "${SAVE_PATCH_IMAGES:-1}"
    done
    ;;
  pdac|gist)
    S1167="${REPO}/data/CODEX/HCC/Michael_data_transfer/s1167"
    if [[ "${COHORT}" == "pdac" ]]; then
      CODE_DIR="${REPO}/code/CODEX_pdac"
      STARDIST_ROOT="${STARDIST_ROOT:-${REPO}/data/CODEX/HCC/StarDist_Segment_pdac/pdac_result}"
    else
      CODE_DIR="${REPO}/code/CODEX_gist"
      STARDIST_ROOT="${STARDIST_ROOT:-${REPO}/data/CODEX/HCC/StarDist_Segment_gist/gist_result}"
    fi
    mapfile -t ACQS < <("${PYTHON}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${CODE_DIR}")
from match_codex_cells_with_pixel import list_aligned_annotated_regions
df = list_aligned_annotated_regions()
for aid in df["ACQUISITION_ID"].astype(str).str.strip():
    print(aid)
PY
)
    if (( ${#ACQS[@]} == 0 )); then
      echo "ERROR: no annotated ${COHORT} cores" >&2
      exit 1
    fi
    for acq in "${ACQS[@]}"; do
      if [[ -n "${ACQ_ID:-}" && "${acq}" != "${ACQ_ID}" ]]; then continue; fi
      he="${S1167}/${acq}/figures/${acq}.tif"
      if [[ "${COORD}" == "gt" ]]; then
        pos="${S1167}/${acq}/${acq}.cell_data.csv"
        sub="ImgEmbeddings_all"
      else
        pos="${STARDIST_ROOT}/${acq}/${acq}_Float_prob0.01_nms_0.3.csv"
        sub="ImgEmbeddings_all_stardist"
      fi
      run_one "sc_${acq}" "${pos}" "${he}" False "" \
        "${S1167}/${acq}/project_all_UNI2/${sub}" "${SAVE_PATCH_IMAGES:-1}"
    done
    ;;
  gbm)
    GBM="${REPO}/data/CODEX/GBM"
    STARDIST_ROOT="${STARDIST_ROOT:-${GBM}/WangLab/StarDist_Segment}"
    for sample in ${SAMPLE:-P174511_Initial P179161_Recurrent}; do
      case "${sample}" in
        P174511_Initial) key="174511-2-3"; he="${GBM}/WangLab/4_Images/Microscope_Image/174511-2-3.tif" ;;
        P179161_Recurrent) key="179161-4-3"; he="${GBM}/WangLab/4_Images/Microscope_Image/179161-4-3.tif" ;;
        *) echo "ERROR: SAMPLE must be P174511_Initial or P179161_Recurrent" >&2; exit 1 ;;
      esac
      if [[ "${COORD}" == "gt" ]]; then
        pos="${GBM}/Cases/${sample}/${sample}_cells_with_pixel.csv"
        sub="ImgEmbeddings_all"
      else
        pos="${STARDIST_ROOT}/${key}/${key}_Float_prob0.01_nms_0.3.csv"
        sub="ImgEmbeddings_all_stardist"
      fi
      run_one "sc_${sample}" "${pos}" "${he}" False "" \
        "${GBM}/Cases/${sample}/project_all_UNI2/${sub}" "${SAVE_PATCH_IMAGES:-1}"
    done
    ;;
  *)
    echo "ERROR: unknown cohort ${COHORT}" >&2
    exit 1
    ;;
esac

echo
echo "=== UNI2 patch_size=14 ${COHORT} ${COORD} gpu=${GPU}: ok=${OK} skip=${SKIP} fail=${FAIL} ==="
if (( FAIL > 0 )); then
  exit 1
fi
```

These commands only extract embeddings. The trainers still read `project_all_UNI/.../sc_pth_16_16` and a 1024-d vector, so this run does not replace the published UNI checkpoint.

## Changelog

### 2026-11-11 — Per-nucleus L2 confidence (`prediction_confidence.py`)

- Added [`prediction_confidence.py`](prediction_confidence.py): UTOPIA-style
  conformal presence plus softmax margin/entropy and similar-type collapse
  (B/Plasma, T subtypes, myeloid, epithelial). Calibrates on pan-cancer OOF
  (`oof_predictions.npz`); scores existing `{sample}_l2_probs.npz` without a
  new train or GPU predict.
- `annotate_nuclei` writes `label_plot` (`pred_CT`, parent family, or
  `low_confidence`). `conformal_from_oof` writes per-class raw-p
  `confidence` and an FPR `present` mask vs the OOF-null quantile
  (`--fdr 0.10` = 90th percentile of non-class cells). Whole-slide BH
  `p_adj` is diagnostic only: it is empty on ~1e5 NPC nuclei because
  UTOPIA BH is per histology cluster and OOF false-positive softmax
  already reaches ~1. `attach_to_obs` copies those columns onto pred `obs`.
- CLI sidecar `{sample}_confidence.npz`. Notebook maps call
  `plot_bcell_presence_figure`, `plot_class_argmax_vs_gated`, and
  `plot_trusted_spatial` in this module. Unit tests in
  [`test_prediction_confidence.py`](test_prediction_confidence.py).

### 2026-10-26 — UNI2 gt and StarDist commands listed in the extractor

- `run_uni2_patch14.sh` now comments both run lists at the top: `gt` for the seven training cohorts (`ImgEmbeddings_all/sc_pth_14_14`) and `stardist` for every segmented nucleus (`ImgEmbeddings_all_stardist/sc_pth_14_14`). StarDist lines set `SAVE_PATCH_IMAGES=0`.
- The same lists are in the UNI2 section above. GT tokens are not a substitute for StarDist centroids. BRCA all-nuclei maps in `code/Xenium_brca/BRCA_pred_uni2_pretrained.ipynb` wait on `brca stardist`.

### 2026-10-25 — Next embedder is UNI2-h

- Recorded why UNI2-h is the encoder to add while `data/` still holds only UNI. H-optimus-1’s required 224 px tile, Virchow2’s flattened token bag, and HIPT’s single 384-d token do not match the per-nucleus MLP.
- Added `run_uni2_patch14.sh` in this README for the seven training cohorts: same slides and coordinates as UNI, `--method UNI2`, `--patch_size 14`, output under `project_all_UNI2/.../sc_pth_14_14`.
- The third argument is the physical GPU. The script activates the SeededNTM conda env and sets `CUDA_VISIBLE_DEVICES` to that index.

### 2026-10-21 — HCC Nature mixed TIFFs in `external_validation`

- `_SIMPLE_COHORTS["HCC_NATURE"]` points at `data/Visium/HCC/nature_data`.
- Per-slide µm/px: plausible OME `PhysicalSize`, else Aperio `MPP`, else fallback 0.5.
- OME sizes outside 0.05–5 µm/px (300 dpi placeholders) are ignored.

### 2026-10-08 — Ontology and fold assignment live in the package

- `pancancer_labels.py` and `pancancer_cv.py` moved here from `code/PanCancer`. The PanCancer copies re-export them so existing scripts and notebooks keep `from pancancer_labels import ...`.
- `palette_for` is in `plotting_palettes.py`. `pancancer_viz` imports it. `write_palette_swatch.py` writes the paper swatch from that same map.
- `external_validation.py` no longer puts `code/PanCancer` on `sys.path`.
- Cohort paths, the section index, and the train / predict CLIs stay in `code/PanCancer`.

### 2026-10-08 — Pan-cancer L12 / L2 colors

- `Cell_Type_COLORS_PANCANCER_level0` (L1) is unchanged.
- L1 stays the five lineage colors. L12 and L2 use the parent hue (epithelial warm, T blue, B teal, myeloid ochre, stromal green, endothelial orange, neural purple), with lightness changes inside each family so neighboring classes stay apart. `get_palette("pancancer", ...)` is the only map.

### 2026-10-07 — StarDist external validation

- Added `external_validation.py`. `ome_um_per_px` reads OME `PhysicalSizeX` / `PhysicalSizeY` from the TIFF `ImageDescription` at run time. `ExternalCohort.geometry` turns that into the scale that maps the slide onto 0.5 µm/px (16 px = 8 µm). TIFF 96 dpi tags are not read.
- JPEG slides have no OME block. Pass `mpp_um` (HCC: env `MPP_UM`). `mpp_source` in the geometry dict records which path was used.
- Cohort folders under `code/External_val/` only set the data root and call `ExternalCohort.main`.

### 2026-08-27 — Shared histology niche-index engine

- Added `histology_niche_index.py`: dataset-neutral TLS / SRI / TNI (AUROC load,
  neighborhood scores, cohort Q4 / burden, source-metric grid).
- Register a new organ with `OrganNicheConfig` + `register_organ_config` (GBM /
  BRCA can call this without copying HCC/PDAC modules).
- CODEX HCC / PDAC / GIST keep thin wrappers for paths and clinical I/O.
- Xenium lung ARI / FRI / pathologist code stays in
  `Xenium_lung/histology_derived_niche_index.py`.

### 2026-08-13 — Dataset-aware plotting refactor

- Moved canonical CODEX ESCC, Xenium lung, and CODEX HCC palettes and palette
  resolution into `plotting_palettes.py`.
- Moved reusable DataFrame count/composition transforms and generic bar,
  stacked-composition, and lineage-panel plots into `plotting_utils.py`.
- Kept spatial plotting in `plot.py`; dataset modules now provide only
  paths, I/O, CLI/report orchestration, and compatibility wrappers/re-exports.
- Removed the package-side dynamic import of the Xenium directory. Package
  modules no longer depend on dataset folders.
- Validated dataset/scheme aliases, semantic-tier resolution, deterministic
  fine-label fallbacks, legacy wrapper imports, and representative CODEX ESCC,
  Xenium, and HCC spatial/bar/composition plotting paths. The historical
  `ncrt` plotting ID remains a backward-compatible alias of `codex_escc`.
- Registered explicit five-tier Xenium lung schemes and migrated shared
  UNI-label spatial/ROC defaults to tier-aware canonical schemes.

### 2026-08-12 — Shared UNI-label CV helpers

Canonical cross-dataset UNI-label CV, internal-validation, and StarDist-tier
helpers moved from `Xenium_lung/xenium_uni_nb_helpers.py` to
`uni_label_cv_helpers.py`. The old Xenium module remains a compatibility shim.

## Module index

| Module | Responsibility |
|--------|----------------|
| [`base.py`](base.py) | Coordinate matching, embedding/AnnData/NPZ data preparation, spatial-neighbor indices, model classes, and low-level evaluation utilities. |
| [`model.py`](model.py) | CV training, checkpoint selection/reload, spatial-context fusion, prediction, and model-level validation. |
| [`plot.py`](plot.py) | Dataset-neutral ROC, confusion-matrix, and spatial plotting APIs, including multi-tier prediction plots. |
| [`plotting_palettes.py`](plotting_palettes.py) | Canonical palette registry, dataset/tier normalization, scheme aliases, RGBA conversion, and deterministic palette resolution. |
| [`plotting_utils.py`](plotting_utils.py) | Generic DataFrame count/composition transforms and reusable bar, stacked-composition, and lineage-panel plots. |
| [`histology_niche_index.py`](histology_niche_index.py) | Shared TLS / SRI / TNI engine (`OrganNicheConfig`, neighborhood scores, cohort burden). |
| [`uni_label_cv_helpers.py`](uni_label_cv_helpers.py) | Shared UNI-label CV, internal-validation, and StarDist prediction/report helpers used by dataset pipelines and notebooks. |
| [`prediction_confidence.py`](prediction_confidence.py) | Per-nucleus L2 confidence from existing softmax: similar-type collapse and UTOPIA-style conformal presence calibrated on pan-cancer OOF. |

## Architecture boundary

The package owns:

- dataset-neutral transforms, model logic, metrics, and plots;
- registered palette data and palette/tier resolution;
- stable public APIs shared by multiple datasets.

Dataset folders own:

- local and cohort-specific paths, file discovery, and file I/O;
- sample/acquisition mappings and dataset metadata;
- CLI entry points, batch orchestration, and cohort reports;
- thin compatibility wrappers that preserve historical imports/defaults.

The dependency direction is one-way: dataset folders may import
`Hist2Pheno_pkg`, but package modules must not import or dynamically load
modules from dataset folders.

## Semantic tiers and legacy mappings

New APIs use semantic tier names. Historical head names remain accepted where
the registry can map them unambiguously.

| Normalized tier | CODEX HCC | Xenium lung | CODEX ESCC (historical NCRT naming) |
|-----------------|-----------|-------------|------------------------|
| `fine` | Excel `celltype_level2`, `final_CT`, L2 | `celltype_level2`, `final_CT`, L2 | `celltype`, L2 |
| `intermediate` | Excel `celltype_level1`, `final_sublineage`, L12 | `final_sublineage`, L12 | `celltype_level12`, L12 |
| `lineage` | Not a separate HCC model target; use `coarse` | `final_lineage`, `celltype_level1`, L1 | `celltype_level1`, L1 |
| `coarse` | Excel `celltype_level0`, `final_lineage`, L1 | Palette alias of Xenium lineage | `celltype_level0`, L3 |
| `lineage_bucket` | Not registered | Not registered | `celltype_level01`, L4 |
| `cniche` | Not registered | `CNiche`, L3 | Not used |
| `tniche` | Not registered | `TNiche`, L4 | Not used |

For HCC, historical model head `L1` means the four-class coarse target.
For Xenium, `L1` means lineage. `L2`, `L12`, `L3`, and `L4` map to `fine`,
`intermediate`, `cniche`, and `tniche`, respectively.
For ESCC, all five tiers use the single `codex_escc` scheme. Spatial callers
select L2/L1/L12/L3/L4 with `celltype_col` values `celltype`,
`celltype_level1`, `celltype_level12`, `celltype_level0`, and
`celltype_level01`. ROC callers use the same values through the legacy-named
`ncrt_color_tier` compatibility parameter.

## Dataset IDs and scheme aliases

Canonical dataset IDs are:

- `codex_escc` (aliases: `escc`, `ncrt`; `ncrt` is retained for backward compatibility)
- `xenium_lung` (aliases: `xenium`, `lung`)
- `codex_hcc` (aliases: `codex`, `hcc`)

Supported schemes are:

- `codex_escc` (`ncrt` is a backward-compatible alias)
- `xenium_lung_fine`, `xenium_lung_intermediate`, `xenium_lung_coarse`,
  `xenium_lung_CNiche`, `xenium_lung_TNiche`
- `codex_hcc`, `codex_hcc_fine`, `codex_hcc_intermediate`,
  `codex_hcc_coarse`

| Xenium source/head | Canonical scheme |
|--------------------|------------------|
| `final_CT` / L2 | `xenium_lung_fine` |
| `final_sublineage` / L12 | `xenium_lung_intermediate` |
| `final_lineage` / L1 | `xenium_lung_coarse` |
| `CNiche` / L3 | `xenium_lung_CNiche` |
| `TNiche` / L4 | `xenium_lung_TNiche` |

The resolver is case-insensitive. `xenium`, `xenium_auto`, `xenium_ct`, and
`xenium_lineage` remain backward-compatible aliases; new code should use the
explicit canonical schemes above.

### `pan_organ` (optional, unified caller API)

Most plotting APIs accept either explicit `*_color_scheme` values **or** an
optional `pan_organ` selector (`codex_hcc`, `xenium_lung`, `codex_escc`). When
`pan_organ` is set, callers can omit per-tier scheme strings and let the
package choose the canonical scheme for L2/L1/L12/L3/L4 based on the plotting
context (e.g. `celltype_col` or `ncrt_color_tier`).

Clinical boxplots use the same selector for x-axis group order:

```python
from plotting_palettes import clinical_group_order, clinical_columns_for_pan_organ

clinical_group_order("xenium_lung")
# Status: Control, Disease; Sample_Affect_Pairing: Unaffected → More_Affected

clinical_group_order("codex_hcc")
# Response: Responder, Non_Responder; diagnosis: Pre, Post

clinical_columns_for_pan_organ("codex_hcc")
# ("Response", "diagnosis", "treatment")
```

`histology_derived_niche_index.test_q4_by_clinical_groups` /
`plot_q4_clinical_comparison` and `uni_label_cv_helpers.analyze_stardist_macro_auroc_by_clinical`
accept `pan_organ=` so HCC and Xenium do not share one mixed order dict.

Precedence is:

- explicit `color_overrides` / `class_color_overrides`
- explicit scheme argument (`spatial_color_scheme` / `roc_color_scheme`)
- `pan_organ` + tier/head context
- legacy default

## Plotting examples

Add `code/Hist2Pheno_pkg` to `PYTHONPATH` before importing these modules.

### Resolve a palette

```python
from plotting_palettes import resolve_palette

colors = resolve_palette(
    ["CD4 T cells", "CD8 T cells", "B cells"],
    scheme="codex_hcc_fine",
)
# Values are matplotlib-compatible RGBA tuples.
```

You can also route through `pan_organ` when you do not want to specify a
scheme explicitly:

```python
colors = resolve_palette(
    ["AT1", "AT2"],
    pan_organ="xenium_lung",
    tier="fine",
)
```

For an incomplete fine-label subset, pass the full training order through
`canonical_labels` to keep fallback colors stable across samples:

```python
colors = resolve_palette(
    labels_present,
    dataset="xenium_lung",
    tier="fine",
    canonical_labels=global_l2_class_names,
)
```

### Spatial plotting

```python
from plot import plot_celltype_spatial_distribution

plot_celltype_spatial_distribution(
    cells,
    x_col="X_pix_HE",
    y_col="Y_pix_HE",
    celltype_col="final_CT",
    spatial_color_scheme="xenium_lung_fine",
    save_path="figures/final_ct.jpg",
    show=False,
)
```

Use `codex_hcc_fine`, `codex_hcc_intermediate`, or `codex_hcc_coarse` for HCC
tiers. `plot_tier_spatial_distribution` is available for encoded prediction
arrays and optional ground-truth panels.

To plot confidence instead of argmax, replace `celltype_col` with
`label_plot` from `annotate_nuclei`, or subset to
`obs["present_B_cell_fdr0.10"]` from `attach_to_obs` (see
[Per-nucleus cell-type confidence](#2026-11-11-per-nucleus-cell-type-confidence)).

### Generic bars and composition plots

```python
from plotting_palettes import resolve_palette
from plotting_utils import (
    celltype_counts_from_df,
    plot_celltype_count_bars,
    plot_celltype_proportions_stacked,
)

counts = celltype_counts_from_df(cells, celltype_col="final_CT")
colors = resolve_palette(counts.index, dataset="xenium_lung", tier="fine")
plot_celltype_count_bars(
    counts,
    celltype_colors=colors,
    save_path="figures/cell_counts.pdf",
    show=False,
)

# Rows are cell types; columns are samples; values are counts.
plot_celltype_proportions_stacked(
    counts_by_sample,
    celltype_colors=resolve_palette(
        counts_by_sample.index, dataset="xenium_lung", tier="fine"
    ),
    save_path="figures/composition.pdf",
    show=False,
)
```

## Compatibility wrappers

Legacy dataset modules may re-export package constants/functions or wrap them
to preserve established names, signatures, and dataset-specific defaults.
These wrappers should be thin: new dataset-neutral behavior belongs in the
package, while path lookup, I/O, CLI behavior, and report composition stay in
the dataset module. New code should import package APIs directly.

Compatibility re-exports are not a second source of truth. Palette values,
semantic-tier rules, and generic plotting implementations must be changed only
in `plotting_palettes.py` or `plotting_utils.py`.

## Registering a future dataset

1. Add its canonical ID and aliases to `_DATASET_ALIASES` in
   `plotting_palettes.py`.
2. Register optional user-facing schemes and default tiers in
   `_SCHEME_DEFAULTS`.
3. Add canonical palette mappings to `get_palette()` and tier inference order
   to `_infer_tier()`.
4. Extend `normalize_tier()` only for genuine legacy column/head aliases.
5. Exercise `resolve_palette()` with every supported tier, unknown labels,
   deterministic fine-label fallback, and explicit overrides.
6. Keep all palette data in the package. The dataset folder may import and
   re-export it, but the package must never import the dataset folder.

This keeps dependency direction stable and allows a new dataset to use shared
plots without coupling package imports to its local directory layout.
