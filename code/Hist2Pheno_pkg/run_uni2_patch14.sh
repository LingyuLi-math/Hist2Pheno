#!/usr/bin/env bash
# UNI2-h, patch_size=14, one class token per nucleus.
# Mirrors the UNI extractors. Writes project_all_UNI2 / sc_pth_14_14.
# Does not read or delete project_all_UNI.
#
# Run from the repo root. The last argument is the physical GPU. SeededNTM is activated inside.
# A cohort whose sc_pth_14_14/ already has .pth files is skipped.
#
#####################################################
# 2026.10.26 LLY - gt for brca
#####################################################
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
#####################################################
# 2026.10.26 LLY - stardist for brca
#####################################################
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
    # 2026.10.28: old .pth count/log check is replaced by completed provenance.
    # A crashed extract leaves .pth files and a log with no "Feature extraction time:".
    local n
    if n="$("${PYTHON}" "${REPO}/code/check_embeddings.py" "${pth_dir}")"; then
      echo "SKIP done (${n} embeddings): ${pth_dir#${REPO}/}"
      SKIP=$((SKIP + 1))
      return 0
    fi
    echo "REDO incomplete/stale embeddings: ${pth_dir#${REPO}/}"
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
