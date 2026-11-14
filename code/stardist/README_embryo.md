# Embryo S046 H&E nuclei segmentation

`run_embryo.sh` defaults to the S046 OME-TIFF in `Embryo_data/tic-4644/data/HnE`.
It uses the bundled `models/2D_versatile_he` model and the existing `hist2pheno`
environment. It reads level-0 image tiles through OpenSlide, with no whole-slide
image array or mask allocation. Nuclear results are written incrementally.
The existing `run_tiff.py` and benchmark scripts are unchanged.

The current `hist2pheno` environment is missing OpenCV (`cv2`), required by the
shared contour extractor. Install it yourself before inference (not installed by
this demo preparation). Keep NumPy 1.x for the existing TensorFlow 2.15 stack:

```bash
/software/conda/users/ll20/envs/hist2pheno/bin/python -m pip install 'numpy<2' 'opencv-python-headless>=4.8,<4.12'
```

Run from the repository root, on your chosen compute node:

```bash
# Metadata / input / model check only: no TensorFlow import or inference.
bash code/SpatialBM/code/stardist/run_embryo.sh --check-only

# Small regional trial: choose X/Y inside tissue, in original slide pixels.
# Replace 10000/10000 with the location you want to inspect.
OUT_DIR="$PWD/Embryo_output/embryo_output/stardist/S046_trial" \
  bash code/SpatialBM/code/stardist/run_embryo.sh --roi 10000 10000 4096 4096

# Full slide. Select a GPU allocated to your job; remove CUDA_VISIBLE_DEVICES
# if your scheduler already sets it.
CUDA_VISIBLE_DEVICES=0 bash code/SpatialBM/code/stardist/run_embryo.sh
```

Default output: `Embryo_output/embryo_output/stardist/S046/`.
The output directory must be empty or absent. Use a new `OUT_DIR` for another
trial; interrupted output is not automatically resumed. Only a final
`run_meta.json` with `status=complete` marks a successfully completed run.
A completed ROI or `--max-tiles N` trial has `partial=true` in its metadata.
No trial was executed during preparation of these scripts.

## Parameters and dependencies

- Default tiles: 2048 px; overlap: 256 px. Lower `TILE=1024` on limited GPU memory.
- Default thresholds come from the bundled model: probability about 0.69248,
  NMS 0.3. Override with `--prob 0.5 --nms 0.3` after inspecting a trial.
  The original benchmark's probability 0.01 is not used as the default here.
- `--roi X Y WIDTH HEIGHT`: limit processing to a region. Coordinates remain
  in the full-resolution slide frame; objects at ROI edges may be truncated.
- `--max-tiles N`: process the first N planned tiles only; these may be blank.
- `--mpp VALUE`: supply a known microns-per-pixel value if slide metadata lacks it.
  Missing scale stays null; all primary coordinates and areas remain pixels.
- `STARDIST_ENV=/path/to/env`, `IMAGE=/path/to/image.ndpi`, and
  `OUT_DIR=/path/to/new/results` override the shell defaults. TIFF and NDPI
  must be formats supported by OpenSlide; unsupported TIFF layouts fail explicitly.
- Python dependencies: numpy, tensorflow, stardist, csbdeep, openslide-python,
  openslide-bin, Pillow, scikit-image, OpenCV (`opencv-python` or
  `opencv-python-headless`), pyarrow. The bundled model weights are used locally.
  `pip-freeze.txt` records the original benchmark environment, not a requirement
  to upgrade this project's existing environment.

## Outputs

- `nuclei_coordinates.csv`: ID, centre x/y, nuclear area (pixels).
- `instances.parquet`: centres, pixel area, polygon area, bounding boxes.
- `instances.geojson`: nuclear boundary polygons and matching measurements.
- `contract.json`: coordinate conventions and x/y physical pixel scale.
- `tile_times.csv`: per-tile runtime and retained nucleus count.
- `run_meta.json`: parameters, partial/full-slide scope, counts, versions and runtime.

Origin is top-left; x is column, y is row. Overlap ownership uses each nucleus's
centroid to reduce duplicates across tiles; predictions can still differ at seams.
Degenerate contours have CSV/Parquet rows but no GeoJSON polygon.
This model segments nuclei; it does not assign cell types, whole-cell boundaries,
or the anatomical level1/level2 labels. Those need downstream analysis.
No automatic background exclusion is used, so whole-slide runs include glass tiles.
Memory also includes model activations and OpenSlide's decoder/cache; streaming
avoids allocating the entire decoded slide but does not guarantee a fixed RAM use.

## Optional batch example

After inspecting the S046 result, process additional selected slides explicitly:

```bash
for section in 046 061; do
  image="$PWD/Embryo_data/tic-4644/data/HnE/CS17_HnE_SOB26-WEM-0-FFPE-1-S${section}.ome.tif"
  [[ -f "$image" ]] || { echo "Missing $image" >&2; exit 1; }
  IMAGE="$image" OUT_DIR="$PWD/Embryo_output/embryo_output/stardist/S${section}" \
    bash code/SpatialBM/code/stardist/run_embryo.sh || exit $?
done
```

Adjust filenames/extensions to actual slides (some sections are `.ndpi`).
An existing output directory causes an error rather than overwriting a prior run.
