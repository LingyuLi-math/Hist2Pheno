import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob


## for AI edit
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def _load_matched_features(matched_features_path):
    from base import load_matched_features_bundle

    return load_matched_features_bundle(matched_features_path)


def _filter_indices_for_class_names(y_true, y_pred, class_names):
    """Keep only samples whose label/pred indices are valid for ``class_names``."""
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    max_idx = len(class_names) - 1
    valid = (y_true >= 0) & (y_true <= max_idx) & (y_pred >= 0) & (y_pred <= max_idx)
    if not np.all(valid):
        n_bad = int((~valid).sum())
        print(
            f"  ⚠ Excluding {n_bad} samples with out-of-range labels/preds "
            f"(valid class indices: 0–{max_idx})."
        )
    return y_true[valid], y_pred[valid]


def _logits_l1_head_from_model(model, x, neighbor_x=None):
    """L1 logits from dual-head or five-head (via ``forward_heads_l2_l1``)."""
    from base import forward_heads_l2_l1

    _, logits_l1 = forward_heads_l2_l1(model, x, neighbor_x=neighbor_x)
    return logits_l1


def _spatial_neighbor_x_for_batch(X_full_t, nbr_idx_t, device, row_start, x_batch):
    if X_full_t is None or nbr_idx_t is None:
        return None
    import torch
    from base import gather_neighbor_embeddings

    b = x_batch.shape[0]
    global_idx = torch.arange(row_start, row_start + b, dtype=torch.long, device=device)
    return gather_neighbor_embeddings(X_full_t, nbr_idx_t, global_idx, device=device)


def plot_celltype_distribution_all(celltype_df, celltype_col='celltype', ax=None, title=None):
    """
    绘制cell type分布柱状图
    
    参数:
    celltype_df: DataFrame, 包含celltype列的数据框
    celltype_col: str, celltype列名，默认为'celltype'
    ax: matplotlib.axes, 可选的子图axes，如果提供则在子图中绘制
    title: str, 可选的标题，如果提供则使用，否则自动生成
    """
    # 统计各类 celltype 数量
    counts = celltype_df[celltype_col].value_counts()
    
    # 如果提供了ax，则在子图中绘制；否则创建新图
    if ax is None:
        plt.figure(figsize=(12,6))
        ax = plt.gca()
        is_subplot = False
        print(f"The number of cells in {celltype_col}: {len(celltype_df[celltype_col])}")
        print(f"The distribution of {celltype_col}: {counts}")
    else:
        is_subplot = True
    
    # 柱状图可视化
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_ylabel('Cell Count')
    
    # 设置标题
    if title is None:
        title = f'Celltype Distribution'
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    if not is_subplot:
        plt.tight_layout()
        plt.show()
    else:
        # 如果是子图，调整布局
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    return counts


def plot_celltype_distribution(celltype_df, celltype_col='celltype', format='pdf', save_path=None):

    counts = celltype_df[celltype_col].value_counts()
    print(f"The number of cells in {celltype_col}: {len(celltype_df[celltype_col])}")
    print(f"The distribution of {celltype_col}: {counts}")

    plt.figure(figsize=(12,6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.ylabel('Cell Count')
    plt.title(f'Celltype Distribution of {celltype_col}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, transparent=True, format=format, dpi=300, bbox_inches='tight')
    else:
        plt.show()


import re
def extract_number(filename):
    ## Extract the number from the filename, e.g. NCRT_tumor12.csv -> 12
    basename = os.path.splitext(os.path.basename(filename))[0]
    nums = re.findall(r'\d+', basename)
    return int(nums[-1]) if nums else 0





def visualize_all_celltype_distributions(base_path='Collaborate/esccAI/data/codex_celltype', 
                                         n_cols=4, n_rows=7, figsize=(28, 28)):
    """
    Batch visualization of cell type distributions for all CSV files in each folder under codex_celltype

    Parameters:
    base_path: str, path to the codex_celltype folder
    n_cols: int, number of subplot columns, default is 4
    n_rows: int, number of subplot rows, default is 7
    figsize: tuple, figure size, default is (20, 28)
    """


    # Get all folders
    folders = [f for f in os.listdir(base_path) 
               if os.path.isdir(os.path.join(base_path, f)) and not f.startswith('.')]
    folders.sort()

    print(f"Found {len(folders)} folders: {folders}")

    # Process each folder
    for folder_name in folders:
        folder_path = os.path.join(base_path, folder_name)

        # Get all CSV files in the folder
        # csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
        
        # Get all CSV files in the folder and sort by number
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        csv_files = sorted(csv_files, key=extract_number)

        if len(csv_files) == 0:
            print(f"Warning: No CSV files in folder {folder_name}")
            continue

        print(f"\nProcessing folder: {folder_name} ({len(csv_files)} files)")

        # Create PDF file
        pdf_path = os.path.join(base_path, f'{folder_name}_celldist.pdf')

        with PdfPages(pdf_path) as pdf:
            # Create figure and subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            # Process each CSV file
            for idx, csv_file in enumerate(csv_files):
                if idx >= len(axes):
                    break

                try:
                    # Read CSV file
                    df = pd.read_csv(csv_file)

                    # Use filename (without extension) as title
                    file_name = os.path.splitext(os.path.basename(csv_file))[0]

                    # Plot in the corresponding subplot
                    ax = axes[idx]
                    plot_celltype_distribution_all(df, celltype_col='celltype', ax=ax, title=file_name)

                except Exception as e:
                    print(f"  Error processing file {csv_file}: {e}")
                    axes[idx].text(0.5, 0.5, f'Error loading\n{os.path.basename(csv_file)}', 
                                  ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].set_title(os.path.basename(csv_file), fontsize=8)

            # Hide redundant subplots
            for idx in range(len(csv_files), len(axes)):
                axes[idx].axis('off')

            # Adjust layout
            plt.tight_layout()

            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        print(f"  Saved: {pdf_path}")

    print(f"\nAll visualizations completed! PDF files are saved in: {base_path}")


# Level0 spatial (coarse compartment; codex_meta_celltype_level012.csv).
_LEVEL0_SPATIAL_DISTINCT_RGBA = {
    "Immune": (82/255, 182/255, 174/255, 1.0),     # #52B6AE
    "Stromal": (240/255, 180/255, 70/255, 1.0),    # #F0B446
    "Tumor": (255/255, 248/255, 0/255, 1.0),       # #FFF800
    "Epithelial": (114/255, 9/255, 101/255, 1.0),  # #720965
}

# Level01 spatial (lineage bucket: B / T / Stromal / Myeloid / Tumor / Epithelial).
_LEVEL01_SPATIAL_DISTINCT_RGBA = {
    "B": (234/255, 107/255, 168/255, 1.0),         # #EA6BA8
    "T": (47/255, 14/255, 181/255, 1.0),           # #2F0EB5
    "Stromal": (240/255, 180/255, 70/255, 1.0),    # #F0B446
    "Myeloid": (82/255, 182/255, 174/255, 1.0),    # #52B6AE, same as level1 Myeloid
    "Immune": (82/255, 182/255, 174/255, 1.0),     # legacy alias for older level012 CSVs
    "Tumor": (255/255, 248/255, 0/255, 1.0),       # #FFF800
    "Epithelial": (114/255, 9/255, 101/255, 1.0),  # #720965
}

# Level1 spatial / summary (user hex → RGBA in [0,1] for matplotlib).
_LEVEL1_SPATIAL_DISTINCT_RGBA = {
    "Epithelial": (114/255, 9/255, 101/255, 1.0),  # #720965
    "Tumor": (255/255, 248/255, 0/255, 1.0),       # #FFF800
    "Stromal": (240/255, 180/255, 70/255, 1.0),    # #F0B446
    "Myeloid": (82/255, 182/255, 174/255, 1.0),    # #52B6AE
    "CD8+T": (47/255, 14/255, 181/255, 1.0),       # #2F0EB5
    "CD4+T": (5/255, 247/255, 15/255, 1.0),        # #05F70F
    "B": (234/255, 107/255, 168/255, 1.0),         # #EA6BA8
}

# celltype_level12 buckets (codex_meta_celltype_level012.csv); distinct from tab20 fallbacks.
_LEVEL12_SPATIAL_DISTINCT_RGBA = {
    "B_other": (234/255, 107/255, 168/255, 1.0),      # #EA6BA8, B-lineage
    "B_plasma": (196/255, 69/255, 140/255, 1.0),      # #C4458C
    "CD4_Tconv": (5/255, 247/255, 15/255, 1.0),       # #05F70F
    "CD4_Treg": (10/255, 143/255, 60/255, 1.0),       # #0A8F3C
    "CD8_Other": (47/255, 14/255, 181/255, 1.0),      # #2F0EB5
    "CD8_Effector": (26/255, 188/255, 156/255, 1.0),  # #1ABC9C
    "Stromal_other": (240/255, 180/255, 70/255, 1.0), # #F0B446
    "Macrophage": (82/255, 182/255, 174/255, 1.0),    # #52B6AE
    "Dendritic": (142/255, 68/255, 173/255, 1.0),     # #8E44AD
    "Myeloid_other": (230/255, 126/255, 34/255, 1.0), # #E67E22
    "Tumor": (255/255, 248/255, 0/255, 1.0),          # #FFF800
    "Epithelial": (114/255, 9/255, 101/255, 1.0),     # #720965
}


# Fine-grained NCRT celltype list (stable tab20 assignment order; same as spatial plots).
_NCRT_FINE_CELLTYPE_ORDER = [
    "Treg", "Bn", "B_other", "CD4_Tn", "Endothelial", "CAF_other",
    "Bm_switched", "CD8_Tn", "Tfh_CXCL13", "CAF_ap", "pDC", "CD4_Tex",
    "Th17", "Macro_other", "B_proliferating", "CD4_Tcm", "Neutrophil",
    "Tfh_CXCR5", "Tfh_CXCL13_CXCR5", "HEV", "Monocyte", "Muscle&mCAF",
    "CD8_Trm", "CD4_Tem", "CD8_Tpex", "Plasma", "CD8_Tem", "CD8_Teff",
    "DC_mature", "cDC2", "cDC1", "CD8_Trm_ex", "CD8_Tex",
    "non_specific", "Macro_M1", "Low_quality", "Epithelial",
    "Tumor_PDL1pos_MHCIpos", "Tumor_PDL1neg_MHCIpos", "Bm_unswitched",
    "Tumor_PDL1neg_MHCIneg", "Tumor_PDL1pos_MHCIneg",
    "Epithelial", "Stromal", "Myeloid", "CD8+T", "Tumor", "CD4+T", "B",
]


def build_ncrt_spatial_color_map(celltype_col="celltype"):
    """
    Label -> RGBA map used by ``plot_celltype_spatial_distribution`` (``spatial_color_scheme='ncrt'``).

    ``celltype_col`` selects the tier palette: ``celltype``, ``celltype_level1``,
    ``celltype_level12``, ``celltype_level0``, or ``celltype_level01``.
    """
    import numpy as np

    if celltype_col == "celltype_level0":
        return dict(_LEVEL0_SPATIAL_DISTINCT_RGBA)
    if celltype_col == "celltype_level01":
        return dict(_LEVEL01_SPATIAL_DISTINCT_RGBA)
    if celltype_col == "celltype_level1":
        return dict(_LEVEL1_SPATIAL_DISTINCT_RGBA)
    if celltype_col == "celltype_level12":
        return dict(_LEVEL12_SPATIAL_DISTINCT_RGBA)

    all_celltypes = list(_NCRT_FINE_CELLTYPE_ORDER)
    level1_custom_colors = dict(_LEVEL1_SPATIAL_DISTINCT_RGBA)
    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors_tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
    colors_set1 = plt.cm.Set1(np.linspace(0, 1, 10))
    all_colors = np.vstack([colors_tab20, colors_tab20b, colors_set1])
    while len(all_colors) < len(all_celltypes):
        all_colors = np.vstack([all_colors, np.array([[0.5, 0.5, 0.5, 1.0]])])
    for i in range(len(all_colors)):
        r, g, b, a = all_colors[i]
        brightness = (r + g + b) / 3.0
        if brightness > 0.75:
            factor = 0.6 if brightness > 0.85 else 0.75
            all_colors[i] = (r * factor, g * factor, b * factor, a)
    fixed_color_map = dict(zip(all_celltypes, all_colors))
    fixed_color_map.update(level1_custom_colors)
    fixed_color_map.update(_LEVEL12_SPATIAL_DISTINCT_RGBA)
    return fixed_color_map


def ncrt_roc_color_overrides(class_names, celltype_col="celltype"):
    """Per-class RGBA list aligned with NCRT spatial colors (for ``class_color_overrides``)."""
    color_map = build_ncrt_spatial_color_map(celltype_col)
    gray = (0.5, 0.5, 0.5, 1.0)
    return {str(nm): tuple(color_map.get(str(nm), gray)[:4]) for nm in class_names}



########################################################
# 2026.06.24: Xenium spatial color overrides 
########################################################
def _import_xenium_spatial_color_module():
    """Import Xenium lung spatial palette helpers (module was renamed)."""
    import sys
    from pathlib import Path

    _xdir = Path(__file__).resolve().parent.parent / "Xenium_lung"
    if _xdir.is_dir() and str(_xdir) not in sys.path:
        sys.path.insert(0, str(_xdir))
    for mod_name in ("plot_HEanno_spatial_labels", "plot_spatial_he_labels"):
        try:
            return __import__(mod_name)
        except ImportError:
            continue
    raise ImportError("plot_HEanno_spatial_labels not found under code/Xenium_lung/")


def build_xenium_spatial_color_overrides(
    class_names,
    spatial_color_scheme="xenium_ct",
):
    """Fixed label→RGBA map for all training classes (pred/true panels share colors)."""
    scheme = (spatial_color_scheme or "xenium_ct").lower()
    if scheme not in ("xenium", "xenium_auto", "xenium_lineage", "xenium_ct"):
        return None
    try:
        mod = _import_xenium_spatial_color_module()
        tier = "lineage" if scheme == "xenium_lineage" else "ct" if scheme == "xenium_ct" else "auto"
        return mod.resolve_xenium_spatial_color_overrides(
            class_names,
            tier=tier,
            canonical_labels=class_names,
        )
    except Exception as exc:
        print(f"  ⚠ Could not build Xenium spatial colors ({exc}); using NCRT/default map.")
        return None


def _resolve_spatial_color_overrides(
    unique_celltypes,
    color_overrides,
    spatial_color_scheme,
    canonical_labels=None,
):
    """Merge explicit overrides with NCRT defaults or Xenium palettes (``plot_HEanno_spatial_labels``)."""
    scheme = (spatial_color_scheme or "ncrt").lower()
    if scheme == "ncrt":
        return color_overrides

    xenium_overrides = None
    if scheme in ("xenium", "xenium_auto", "xenium_lineage", "xenium_ct"):
        try:
            mod = _import_xenium_spatial_color_module()
            tier = "lineage" if scheme == "xenium_lineage" else "ct" if scheme == "xenium_ct" else "auto"
            xenium_overrides = mod.resolve_xenium_spatial_color_overrides(
                unique_celltypes,
                tier=tier,
                canonical_labels=canonical_labels,
            )
        except Exception as exc:
            print(f"  ⚠ Could not load Xenium spatial colors ({exc}); using NCRT/default map.")

    if xenium_overrides and color_overrides:
        merged = dict(xenium_overrides)
        merged.update(color_overrides)
        return merged
    return xenium_overrides or color_overrides
    
########################################################

def _roc_colors_for_class_names(
    class_names,
    roc_color_scheme="ncrt",
    class_color_overrides=None,
    ncrt_color_tier="celltype",
):
    """
    Per-class RGBA colors for ROC curves (index ``i`` = ``class_names[i]``).

    ``ncrt`` uses the same label->RGBA map as ``plot_celltype_spatial_distribution``
    (``build_ncrt_spatial_color_map``). ``xenium_ct`` / ``xenium_lineage`` match Xenium spatial.
    """
    import numpy as np

    names = [str(c) for c in class_names]
    n = len(names)
    overrides = class_color_overrides
    scheme = (roc_color_scheme or "ncrt").lower()

    if overrides is None and scheme == "ncrt":
        overrides = ncrt_roc_color_overrides(names, celltype_col=ncrt_color_tier or "celltype")

    if overrides:
        out = []
        for nm in names:
            t = overrides.get(nm, (0.5, 0.5, 0.5, 1.0))
            t = tuple(t)
            if len(t) == 3:
                out.append((float(t[0]), float(t[1]), float(t[2]), 1.0))
            else:
                out.append(tuple(float(x) for x in t[:4]))
        return out

    if scheme in ("xenium", "xenium_auto", "xenium_ct", "xenium_lineage"):
        sch = "xenium_auto" if scheme == "xenium" else scheme
        xenium_overrides = _resolve_spatial_color_overrides(names, None, sch)
        if xenium_overrides:
            return [
                tuple(xenium_overrides[nm][:4]) if nm in xenium_overrides else (0.5, 0.5, 0.5, 1.0)
                for nm in names
            ]

    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors_tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
    pool = np.vstack([colors_tab20, colors_tab20b])
    return [tuple(pool[i % len(pool)]) for i in range(n)]


def plot_celltype_spatial_distribution(
    celltype_df,
    x_col='Centroid X µm',
    y_col='Centroid Y µm',
    celltype_col='celltype',
    figsize=(14, 10),
    alpha=0.6,
    s=1,
    format='pdf',
    save_path=None,
    color_overrides=None,
    spatial_color_scheme="ncrt",
    title=None,
    show=True,
):
    """
    Plot the spatial distribution of cell types.
    
    Parameters:
    celltype_df: DataFrame, a dataframe containing coordinate and celltype columns
    x_col: str, name of the X coordinate column, default is 'Centroid X µm'
    y_col: str, name of the Y coordinate column, default is 'Centroid Y µm'
    celltype_col: str, annotation column to color by (e.g. ``celltype``, ``celltype_level12``,
        ``celltype_level1``, ``celltype_level01``, ``celltype_level0``). Default ``celltype``.
    figsize: tuple, figure size, default is (14, 10)
    alpha: float, point transparency, default is 0.6
    s: float, point size, default is 1
    color_overrides: dict[str, tuple] optional mapping celltype name -> RGBA (r,g,b,a) in [0,1];
        applied after the default tab20 map so level1 plots can use maximally distinct hues.
    spatial_color_scheme: ``ncrt`` (default, unchanged NCRT behavior), ``xenium_lineage``,
        ``xenium_ct``, or ``xenium_auto`` (lineage colors for 4-class labels else tab20 CT colors).
    title: optional plot title (default ``{celltype_col} Spatial Distribution``).
    show: call ``plt.show()`` when True; set False when saving only (avoids duplicate Jupyter display).

    Color maps are chosen from ``celltype_col`` when it matches annotation tiers:
    ``celltype_level0``, ``celltype_level01``, ``celltype_level1``, or fine-grained
    ``celltype_level12`` / ``celltype`` (tab20-based list + level1 / level12 overrides).
    """
    import numpy as np

    fixed_color_map = build_ncrt_spatial_color_map(celltype_col)

    # Remove rows with missing coordinates
    celltype_df_clean = celltype_df.dropna(subset=[x_col, y_col])
    print(f"Number of valid data points: {len(celltype_df_clean)}")
    print(f"Number of cell types: {celltype_df_clean[celltype_col].nunique()}")
    
    # Get all unique cell types in the data and count cells for each type
    celltype_counts = celltype_df_clean[celltype_col].value_counts()
    unique_celltypes = celltype_counts.index.tolist()  # Already sorted by count (descending)
    
    # Use fixed color mapping for cell types that exist in the data
    # For any cell types not in the fixed list, use a default color (gray)
    color_map = {}
    for ct in unique_celltypes:
        if ct in fixed_color_map:
            color_map[ct] = fixed_color_map[ct]
        else:
            color_map[ct] = (0.5, 0.5, 0.5, 1.0)  # Gray color for unknown cell types

    scheme = (spatial_color_scheme or "ncrt").lower()
    if color_overrides and scheme != "ncrt":
        # Caller supplied a fixed label→RGBA map (e.g. all training class_names).
        scheme_overrides = dict(color_overrides)
    else:
        scheme_overrides = _resolve_spatial_color_overrides(
            unique_celltypes,
            color_overrides,
            spatial_color_scheme,
        )
    if scheme_overrides:
        for k, v in scheme_overrides.items():
            if v is not None:
                t = tuple(v)
                if len(t) == 3:
                    color_map[k] = (float(t[0]), float(t[1]), float(t[2]), 1.0)
                else:
                    color_map[k] = tuple(float(x) for x in t[:4])

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot scatter plot for each cell type (already sorted by count in descending order)
    for celltype in unique_celltypes:
        mask = celltype_df_clean[celltype_col] == celltype
        subset = celltype_df_clean[mask]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            c=[color_map[celltype]],
            label=f'{celltype} (n={len(subset)})',
            alpha=alpha,
            s=s,
        )

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    plot_title = title if title is not None else f"{celltype_col} Spatial Distribution"
    ax.set_title(plot_title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10,
              scatterpoints=1, markerscale=6, frameon=False, title='Cell Type Count', ncol=2)  
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, transparent=True, format=format, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


##################################################
# 2026.03.03 LLY PCF vs HE counts
##################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd

def plot_PCF_HE_counts(
    star_coords_in_roi,
    celltype_pixel_NCRT,
    ROI="NCRT_tumor1",
    format="jpg",
    figure_size=(12, 6),
    save_path=None,
    max_scatter_points=500_000,
):
    """
    Plot cells of a given ROI from two coordinate tables side by side
    (STAR coordinates vs celltype_pixel_NCRT coordinates), and return
    a small summary DataFrame called `merged`.

    Parameters
    ----------
    star_coords_in_roi : pd.DataFrame
        DataFrame containing at least the columns:
        - 'TumorID'
        - 'centroid_x'
        - 'centroid_y'
    celltype_pixel_NCRT : pd.DataFrame
        DataFrame containing at least the columns:
        - 'TumorID'
        - 'X_pix_HE'
        - 'Y_pix_HE'
    ROI : str, optional
        TumorID to plot (e.g. 'NCRT_tumor1'). If the string ends with ``_all``
        (e.g. ``SA_all`` from ``parent_value=all``), no TumorID filter is applied
        and the full tables are plotted (axis limits from all rows; scatter may
        subsample for speed).
    max_scatter_points : int
        When using full-slide mode or very large ROIs, randomly subsample to at
        most this many rows per panel before ``scatter`` (limits still use all rows).

    Returns
    -------
    merged : pd.DataFrame
        Summary table with one row for this ROI, including:
        - TumorID
        - n_star_cells     : number of cells in star_coords_in_roi
        - n_pixel_cells    : number of cells in celltype_pixel_NCRT
        - x_min, x_max     : shared x-range used for plotting
        - y_min, y_max     : shared y-range used for plotting
    """

    use_full_slide = isinstance(ROI, str) and ROI.endswith("_all")

    if use_full_slide:
        tumor_star = star_coords_in_roi
        tumor_pixel = celltype_pixel_NCRT
    else:
        tumor_star = star_coords_in_roi[star_coords_in_roi["TumorID"] == ROI]
        tumor_pixel = celltype_pixel_NCRT[celltype_pixel_NCRT["TumorID"] == ROI]

    print(f"{ROI} - star_coords_in_roi shape: {tumor_star.shape}")
    print(f"{ROI} - celltype_pixel_NCRT shape: {tumor_pixel.shape}")

    if tumor_star.empty or tumor_pixel.empty:
        print(
            f"[plot_PCF_HE_counts] Skip plot: empty subset for ROI={ROI!r} "
            "(for parent_value=all use ROI ending with _all, e.g. SA_all)."
        )
        return pd.DataFrame(
            {
                "TumorID": [ROI],
                "n_star_cells": [len(tumor_star)],
                "n_pixel_cells": [len(tumor_pixel)],
                "x_min": [np.nan],
                "x_max": [np.nan],
                "y_min": [np.nan],
                "y_max": [np.nan],
            }
        )

    # Shared axis limits from full subset (before scatter subsampling)
    x_min = min(
        float(tumor_star["centroid_x"].min()),
        float(tumor_pixel["X_pix_HE"].min()),
    )
    x_max = max(
        float(tumor_star["centroid_x"].max()),
        float(tumor_pixel["X_pix_HE"].max()),
    )
    y_min = min(
        float(tumor_star["centroid_y"].min()),
        float(tumor_pixel["Y_pix_HE"].min()),
    )
    y_max = max(
        float(tumor_star["centroid_y"].max()),
        float(tumor_pixel["Y_pix_HE"].max()),
    )

    if not np.isfinite(x_min) or not np.isfinite(x_max) or not np.isfinite(y_min) or not np.isfinite(y_max):
        print(f"[plot_PCF_HE_counts] Skip plot: non-finite axis limits for ROI={ROI!r}")
        return pd.DataFrame(
            {
                "TumorID": [ROI],
                "n_star_cells": [len(tumor_star)],
                "n_pixel_cells": [len(tumor_pixel)],
                "x_min": [x_min],
                "x_max": [x_max],
                "y_min": [y_min],
                "y_max": [y_max],
            }
        )

    star_plot = tumor_star
    pixel_plot = tumor_pixel
    if max_scatter_points is not None and int(max_scatter_points) > 0:
        if len(star_plot) > max_scatter_points:
            star_plot = star_plot.sample(n=int(max_scatter_points), random_state=0)
        if len(pixel_plot) > max_scatter_points:
            pixel_plot = pixel_plot.sample(n=int(max_scatter_points), random_state=0)
    if use_full_slide and (len(star_plot) < len(tumor_star) or len(pixel_plot) < len(tumor_pixel)):
        print(
            f"[plot_PCF_HE_counts] Subsampled scatter to max {max_scatter_points} points per panel "
            f"(limits from full data: star {len(tumor_star)}, pixel {len(tumor_pixel)})."
        )

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharex=True, sharey=True)

    # Left: STAR coordinates
    axes[0].scatter(
        star_plot["centroid_x"],
        star_plot["centroid_y"],
        s=0.05,
        c="red",
        alpha=0.6,
    )
    axes[0].set_title(f"Cells in {ROI} (HE: StarDist)")
    axes[0].set_xlabel("centroid_x")
    axes[0].set_ylabel("centroid_y")
    axes[0].invert_yaxis()  # flip y-axis to match image coordinates

    # Right: celltype_pixel_NCRT coordinates
    axes[1].scatter(
        pixel_plot["X_pix_HE"],
        pixel_plot["Y_pix_HE"],
        s=0.05,
        c="blue",
        alpha=0.6,
    )
    axes[1].set_title(f"Cells in {ROI} (PCF: Ground Truth)")
    axes[1].set_xlabel('pxl_col_in_fullres')
    axes[1].invert_yaxis()

    # Apply the same x/y limits to both subplots
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_max, y_min)  # note: inverted y-axis
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_max, y_min)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, transparent=True, format=format, dpi=300, bbox_inches='tight')
    plt.show()

    # Build a small summary DataFrame to return
    merged = pd.DataFrame({
        'TumorID': [ROI],
        'n_star_cells': [len(tumor_star)],
        'n_pixel_cells': [len(tumor_pixel)],
        'x_min': [x_min],
        'x_max': [x_max],
        'y_min': [y_min],
        'y_max': [y_max],
    })

    # return merged
    return merged


import numpy as np
import matplotlib.pyplot as plt


def _infer_tumor_id_prefix(series):
    """
    Infer therapy prefix like 'NCRT_' or 'SA_' from the first non-null TumorID
    (e.g. SA_tumor3 -> SA_).
    """
    if series is None or len(series) == 0:
        return None
    s = series.dropna().astype(str)
    if len(s) == 0:
        return None
    first = str(s.iloc[0])
    if "_" not in first:
        return None
    return first.split("_", 1)[0] + "_"


def plot_WSI_counts(celltype, roi_cell_counts, prefix=None, figure_size=(8, 5), save_path=None):
    """
    Count cells per TumorID in `celltype` for TumorID starting with a given prefix,
    merge with `roi_cell_counts`, plot grouped bar chart, and return the merged table.

    Additionally, print TumorIDs where the orange bar (roi_cell_counts / HE)
    is lower than the blue bar (celltype / PCF).

    Parameters
    ----------
    prefix : str or None
        Only rows with TumorID starting with this prefix are counted as PCF (blue).
        If None, the prefix is inferred from ``roi_cell_counts`` / ``celltype`` TumorID
        (e.g. ``SA_`` for SA workflows). If you pass ``'NCRT_'`` while the data use
        ``SA_tumor*``, PCF counts will be empty unless you pass ``prefix='SA_'`` or None.
    """

    tid_str = celltype["TumorID"].astype(str)

    # 1. Resolve prefix: explicit, inferred, or legacy default
    if prefix is None:
        prefix = _infer_tumor_id_prefix(roi_cell_counts["TumorID"])
        if prefix is None:
            prefix = _infer_tumor_id_prefix(celltype["TumorID"])
        if prefix is None:
            prefix = "NCRT_"

    mask = tid_str.str.startswith(prefix)
    celltype_ncrt = celltype[mask].copy()

    # Wrong prefix (e.g. NCRT_ with SA_tumor* data) yields no PCF rows — retry once
    if celltype_ncrt.empty and not celltype.empty:
        inferred = _infer_tumor_id_prefix(celltype["TumorID"])
        if inferred and inferred != prefix:
            print(
                f"[plot_WSI_counts] No celltype rows for prefix={prefix!r}; "
                f"using inferred prefix={inferred!r} (match celltype TumorID to HE)."
            )
            prefix = inferred
            mask = tid_str.str.startswith(prefix)
            celltype_ncrt = celltype[mask].copy()

    # 2. Count cells per TumorID in celltype
    celltype_counts = (
        celltype_ncrt['TumorID']
        .value_counts()
        .rename('celltype_count')
        .reset_index()
        .rename(columns={'index': 'TumorID'})
    )

    # 3. Merge with roi_cell_counts on TumorID
    merged = pd.merge(
        celltype_counts,
        roi_cell_counts,          # has columns: TumorID, cell_count
        on='TumorID',
        how='outer'
    )

    # 4. Fill missing counts with 0 and make them integers
    merged[['celltype_count', 'cell_count']] = (
        merged[['celltype_count', 'cell_count']]
        .fillna(0)
        .astype(int)
    )

    # 5. Extract numeric tumor index (e.g. 1, 2, 3, ...) and sort
    merged['tumor_num'] = merged['TumorID'].str.extract(r'(\d+)').astype(int)
    merged = merged.sort_values('tumor_num').reset_index(drop=True)

    # ---- NEW PART: find where orange (cell_count) < blue (celltype_count) ----
    mask_orange_lower = merged['cell_count'] < merged['celltype_count']
    rois_orange_lower = merged.loc[mask_orange_lower, 'TumorID']

    print("ROIs where orange bar (HE: cell_count) is lower than blue bar (PCF: celltype_count):")
    print(rois_orange_lower.to_list())

    # 6. Plot grouped bar chart with the same TumorID as x-axis
    x = np.arange(len(merged))   # positions for TumorID
    width = 0.4                  # width of each bar

    plt.figure(figsize=figure_size)
    
    # Bars for counts from celltype (blue)
    plt.bar(
        x - width/2,
        merged['celltype_count'],
        width=width,
        label='celltype count PCF',
        color='steelblue'
    )

    # Bars for counts from roi_cell_counts (orange)
    plt.bar(
        x + width/2,
        merged['cell_count'],
        width=width,
        label='roi_cell_counts HE',
        color='orange'
    )

    plt.xticks(x, merged['TumorID'], rotation=90)
    plt.ylabel('Cell count')
    plt.xlabel('TumorID')
    plt.title(f'Cell counts per {prefix} tumor (PCF vs HE)')
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, transparent=True, format='pdf', dpi=300, bbox_inches='tight')

    plt.show()

    return merged



##############################################################
# 2026.02.26 LLY Plot cell area histogram
##############################################################
import matplotlib.pyplot as plt
def plot_cell_area_histogram(df, col_name='Cell: Area µm^2', bins=100, figsize=(8, 4), save_path=None):
    """
    Plot a histogram of cell area distribution and display key statistics.

    Parameters:
        df: pandas.DataFrame, containing the cell area data
        col_name: str, the column name for cell area
        bins: int, number of bins for the histogram
    """
    plt.figure(figsize=figsize)

    # Extract cell area data and remove NaN values
    cell_area = df[col_name].dropna()

    # Create histogram
    n, bins, patches = plt.hist(cell_area, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

    # Set title and labels
    plt.xlabel('Cell Area (µm²)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Cell Area (µm²)', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Calculate statistics
    mean_area = cell_area.mean()
    median_area = cell_area.median()
    std_area = cell_area.std()
    q25 = cell_area.quantile(0.25)
    q75 = cell_area.quantile(0.75)

    # Add vertical lines for mean and median
    plt.axvline(mean_area, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_area:.2f} µm²')
    plt.axvline(median_area, color='green', linestyle='--', linewidth=2, label=f'Median: {median_area:.2f} µm²')
    plt.legend(fontsize=10, loc='lower right')

    # Add a text box with statistics
    stats_text = (
        f'Total cells: {len(cell_area):,}\n'
        f'Mean: {mean_area:.2f} µm²\n'
        f'Median: {median_area:.2f} µm²\n'
        f'Std: {std_area:.2f} µm²\n'
        f'Q25: {q25:.2f} µm²\n'
        f'Q75: {q75:.2f} µm²\n'
        f'Min: {cell_area.min():.2f} µm²\n'
        f'Max: {cell_area.max():.2f} µm²'
    )
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, transparent=True, format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()

    # Print summary statistics in terminal
    print(f"\nSummary Statistics for '{col_name}':")
    print(f"  Total cells: {len(cell_area):,}")
    print(f"  Mean: {mean_area:.2f} µm²")
    print(f"  Median: {median_area:.2f} µm²")
    print(f"  Standard deviation: {std_area:.2f} µm²")
    print(f"  Range: [{cell_area.min():.2f}, {cell_area.max():.2f}] µm²")
    print(f"  IQR: [{q25:.2f}, {q75:.2f}] µm²")


##############################################################
# 2026.02.26 LLY Plot confusion matrix
##############################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(
    test_labels,
    test_preds,
    class_names,
    normalize=True,
    figsize=(8, 6),
    cmap="viridis",
    hierarchical=False,
    save_path=None,
):
    """
    Plot confusion matrix with optional hierarchical clustering.

    Args:
        test_labels (array-like): Ground truth labels.
        test_preds (array-like): Predicted labels.
        class_names (list): List of class names.
        normalize (bool): Normalize matrix by row.
        figsize (tuple): Figure size.
        cmap (str): Color map for heatmap.
        save_path (str): Path to save figure.
        hierarchical (bool): If True, apply hierarchical clustering to reorder classes.
    """

    print("Computing confusion matrix...")

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    test_labels, test_preds = _filter_indices_for_class_names(test_labels, test_preds, class_names)
    if len(test_labels) == 0:
        print("  ⚠ No valid label/pred pairs after filtering; skip confusion matrix.")
        return None

    # Only keep classes that actually appear
    unique_labels = np.unique(np.concatenate([test_labels, test_preds]))
    actual_class_names = [class_names[i] for i in unique_labels]

    # Calculate sample count for each class (from test_labels)
    from collections import Counter
    class_counts = Counter(test_labels)

    # Create a list of (label, count, class_name) tuples
    label_info = [(label, class_counts.get(label, 0), actual_class_names[idx]) 
                  for idx, label in enumerate(unique_labels)]

    # Sort by count in descending order
    label_info_sorted = sorted(label_info, key=lambda x: x[1], reverse=True)

    # Extract sorted labels and class names
    sorted_labels = [item[0] for item in label_info_sorted]
    sorted_class_names = [item[2] for item in label_info_sorted]

    # Compute confusion matrix with original order first
    cm = confusion_matrix(test_labels, test_preds, labels=unique_labels)

    # Reorder confusion matrix according to sorted labels
    # Create mapping from original unique_labels to sorted_labels
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    sorted_indices = [label_to_idx[label] for label in sorted_labels]

    # Reorder rows and columns
    cm_sorted = cm[np.ix_(sorted_indices, sorted_indices)]

    if hierarchical:
        # Hierarchical clustering to reorder classes
        print("Applying hierarchical clustering to confusion matrix...")
        from scipy.cluster.hierarchy import linkage, leaves_list

        # Perform clustering on rows (true labels)
        linkage_matrix = linkage(cm_sorted, method="average")
        row_order = leaves_list(linkage_matrix)
        # For symmetric confusion matrix, reorder columns the same way
        cm_sorted = cm_sorted[row_order][:, row_order]
        sorted_class_names = [sorted_class_names[i] for i in row_order]
        sorted_labels = [sorted_labels[i] for i in row_order]

    if normalize:
        # Normalize by row (true label)
        cm_sorted = cm_sorted.astype(float)
        row_sums = cm_sorted.sum(axis=1, keepdims=True)
        # Avoid division by zero
        cm_sorted = np.divide(cm_sorted, row_sums, where=row_sums!=0)

    plt.figure(figsize=figsize)

    sns.heatmap(
        cm_sorted,
        cmap=cmap,
        xticklabels=sorted_class_names,
        yticklabels=sorted_class_names,
        square=True,
        cbar_kws={"label": "Proportion" if normalize else "Count"}
    )

    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


##############################################################
# 2026.03.23 LLY Plot level1 spatial distribution, different from y
##############################################################
def plot_level1_spatial_distribution(
    matched_features_path,
    all_preds,
    class_names_level1,
    class_names,
    y_encoded_f,
    y_level1_encoded_f,
    plot_celltype_spatial_distribution,
    save_path_pred=None,
    save_path_true=None,
    fig_size=(12, 10),
    spatial_point_size=0.6,
    spatial_color_scheme="auto",
    show=True,
    X_coords_matched=None,
    y_level1_f=None,
    title_pred=None,
    title_true=None,
    spatial_title_pred_l1=None,
    spatial_title_true_l1=None,
):
    """
    Plot level1 spatial distribution from all-data predictions.

    Args:
        matched_features_path (str): Path to .npz file containing coordinates and true labels.
        all_preds (array-like): Predicted level2 class indices for all cells.
        class_names_level1 (list): List of level1 class names (for display).
        class_names (list): List of level2 class names (for mapping).
        y_encoded_f (array-like): Level2 encoded labels for mapping (use HE ``cv_data`` encodings;
            do not ``transform`` StarDist ``y_star`` — it may contain CT names absent from training).
        y_level1_encoded_f (array-like): Corresponding level1 encoded labels for mapping.
        plot_celltype_spatial_distribution (function): Plotting function.
        save_path_pred (str or None): Path for saving prediction spatial plot.
        save_path_true (str or None): Path for saving true spatial plot.
        spatial_color_scheme: ``ncrt``, ``xenium_lineage``, or ``auto`` (lineage names → Xenium colors).
        show: display figures in the notebook (still saves when ``save_path_*`` is set).
        X_coords_matched: optional (N,2) array; overrides ``X_coords`` from npz when lengths must
            match ``all_preds`` from ``cv_data`` filtering.
        y_level1_f: optional string level1 labels aligned with ``all_preds`` (e.g. ``cv_data['y_level1_f']``).
        title_pred, title_true: optional plot titles (defaults: Predicted/True level1 spatial distribution).
        spatial_title_pred_l1, spatial_title_true_l1: aliases for ``title_pred`` / ``title_true`` (notebook parity with L2 kwargs).
    """
    title_pred = title_pred or spatial_title_pred_l1
    title_true = title_true or spatial_title_true_l1

    all_preds_l2 = np.asarray(all_preds)
    y_l1_true_for_plot = None

    if X_coords_matched is not None:
        X_coords_all = np.asarray(X_coords_matched, dtype=np.float64)
        n = min(len(X_coords_all), len(all_preds_l2))
        X_coords_all = X_coords_all[:n]
        all_preds_l2 = all_preds_l2[:n]
        if y_level1_f is not None:
            y_l1_true_for_plot = np.asarray(y_level1_f)[:n]
        else:
            y_l1_true_for_plot = np.array(
                [class_names_level1[int(i)] for i in np.asarray(y_level1_encoded_f)[:n]]
            )
    else:
        loaded_data = _load_matched_features(matched_features_path)
        if "X_coords" not in loaded_data:
            raise ValueError(f"X_coords not found in {matched_features_path}")
        X_coords_all = loaded_data["X_coords"]
        n = min(len(X_coords_all), len(all_preds_l2))
        X_coords_all = X_coords_all[:n]
        all_preds_l2 = all_preds_l2[:n]
        if "y_level1" in loaded_data:
            y_l1_raw = loaded_data["y_level1"][:n]
            if y_l1_raw.dtype.kind in ["U", "S", "O"]:
                y_l1_true_for_plot = y_l1_raw
            else:
                y_l1_num = np.asarray(y_l1_raw).astype(np.int64)
                y_l1_true_for_plot = np.array(
                    [
                        class_names_level1[int(i)] if 0 <= int(i) < len(class_names_level1) else f"Unknown_{int(i)}"
                        for i in y_l1_num
                    ]
                )

    # Build level2->level1 mapping (majority vote per L2 class; works for Xenium fine→lineage).
    num_l2_classes = len(class_names)
    child_to_parent = np.full(num_l2_classes, -1, dtype=np.int64)
    y_enc = np.asarray(y_encoded_f)
    y_l1_enc = np.asarray(y_level1_encoded_f)
    for l2_id in range(num_l2_classes):
        mask = y_enc == l2_id
        if not np.any(mask):
            continue
        vals, counts = np.unique(y_l1_enc[mask], return_counts=True)
        child_to_parent[l2_id] = int(vals[int(np.argmax(counts))])
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing level1 mapping for level2 classes: {missing}")

    # Guard against out-of-range predicted level2 indices (e.g. Unknown_39).
    valid_pred_mask = (all_preds_l2 >= 0) & (all_preds_l2 < num_l2_classes)
    if not np.all(valid_pred_mask):
        n_bad = int((~valid_pred_mask).sum())
        bad_vals = np.unique(all_preds_l2[~valid_pred_mask])[:10]
        print(
            f"  ⚠ Found {n_bad} out-of-range level2 predictions for level1 mapping; "
            f"dropping them. Sample bad indices: {bad_vals}"
        )
    X_coords_all = X_coords_all[valid_pred_mask]
    all_preds_l2 = all_preds_l2[valid_pred_mask]

    preds_l1 = child_to_parent[all_preds_l2]
    pred_celltype_l1 = np.array([class_names_level1[int(i)] for i in preds_l1])

    pred_df_l1 = pd.DataFrame({
        'X_pix_HE': X_coords_all[:, 0],
        'Y_pix_HE': X_coords_all[:, 1],
        'celltype': pred_celltype_l1,
    })

    if y_l1_true_for_plot is not None:
        y_true = np.asarray(y_l1_true_for_plot)[valid_pred_mask]
        if y_true.dtype.kind in ["U", "S", "O"] or (
            len(y_true) > 0 and isinstance(y_true.flat[0], (str, np.str_))
        ):
            pred_df_l1["true_celltype"] = y_true
        else:
            y_l1_num = y_true.astype(np.int64)
            pred_df_l1["true_celltype"] = np.array(
                [
                    class_names_level1[int(i)] if 0 <= int(i) < len(class_names_level1) else f"Unknown_{int(i)}"
                    for i in y_l1_num
                ]
            )

    sch = (spatial_color_scheme or "auto").lower()
    if sch == "auto":
        try:
            mod = _import_xenium_spatial_color_module()
            l1_scheme = (
                "xenium_lineage"
                if set(str(c) for c in class_names_level1) <= set(mod.LINEAGE_COLORS.keys())
                else "ncrt"
            )
        except Exception:
            l1_scheme = "ncrt"
    else:
        l1_scheme = sch

    if l1_scheme in ("xenium", "xenium_auto", "xenium_lineage", "xenium_ct"):
        l1_color_overrides = build_xenium_spatial_color_overrides(
            class_names_level1,
            spatial_color_scheme=l1_scheme,
        )
        if l1_scheme == "xenium_lineage":
            print("  Level1 spatial plot: Xenium lineage colors (final_lineage palette).")
    else:
        l1_color_overrides = {
            str(n): _LEVEL1_SPATIAL_DISTINCT_RGBA[n]
            for n in class_names_level1
            if n in _LEVEL1_SPATIAL_DISTINCT_RGBA
        }
        if l1_color_overrides:
            print(f"  Level1 spatial plot: NCRT distinct palette for {len(l1_color_overrides)} class name(s).")

    plot_celltype_spatial_distribution(
        pred_df_l1,
        x_col='X_pix_HE',
        y_col='Y_pix_HE',
        celltype_col='celltype',
        figsize=fig_size,
        alpha=0.6,
        s=spatial_point_size,
        format='jpg',
        save_path=save_path_pred,
        color_overrides=l1_color_overrides or None,
        spatial_color_scheme=l1_scheme,
        title=title_pred or "Predicted level1 spatial distribution",
        show=show,
    )
    if save_path_pred:
        print(f"  Saved level1 predicted spatial plot: {save_path_pred}")

    if 'true_celltype' in pred_df_l1.columns:
        plot_celltype_spatial_distribution(
            pred_df_l1,
            x_col='X_pix_HE',
            y_col='Y_pix_HE',
            celltype_col='true_celltype',
            figsize=fig_size,
            alpha=0.6,
            s=spatial_point_size,
            format='jpg',
            save_path=save_path_true,
            color_overrides=l1_color_overrides or None,
            spatial_color_scheme=l1_scheme,
            title=title_true or "True level1 spatial distribution",
            show=show,
        )
        if save_path_true:
            print(f"  Saved level1 true spatial plot: {save_path_true}")

    # return pred_df_l1


def plot_tier_spatial_distribution(
    pred_encoded,
    class_names,
    plot_celltype_spatial_distribution,
    save_path_pred=None,
    save_path_true=None,
    fig_size=(10, 8),
    spatial_point_size=0.6,
    spatial_color_scheme="xenium_auto",
    celltype_col="celltype",
    show=True,
    X_coords_matched=None,
    y_tier_f=None,
    title_pred=None,
    title_true=None,
):
    """
    Spatial scatter for an encoded label tier (L12 / CNiche / TNiche head predictions).

    Args:
        pred_encoded: integer predictions aligned with ``X_coords_matched`` / ``y_tier_f``.
        class_names: label encoder ``classes_`` for this tier.
        y_tier_f: optional string labels (e.g. ``cv_data['y_level12_f']``) for ground-truth panel.
        spatial_color_scheme: passed to ``plot_celltype_spatial_distribution`` (``xenium_auto`` default).
        celltype_col: NCRT color-tier hint (``celltype_level0`` / ``celltype_level01`` /
            ``celltype_level12``). DataFrame columns remain ``celltype`` / ``true_celltype``.
    """
    import pandas as pd

    _tier_color_maps = {
        "celltype_level0": _LEVEL0_SPATIAL_DISTINCT_RGBA,
        "celltype_level01": _LEVEL01_SPATIAL_DISTINCT_RGBA,
        "celltype_level1": _LEVEL1_SPATIAL_DISTINCT_RGBA,
        "celltype_level12": _LEVEL12_SPATIAL_DISTINCT_RGBA,
    }
    color_overrides = _tier_color_maps.get(celltype_col)

    pred_enc = np.asarray(pred_encoded, dtype=np.int64)
    names = list(class_names)
    n_cls = len(names)

    if X_coords_matched is None:
        raise ValueError("X_coords_matched is required for tier spatial plots.")

    X_coords_all = np.asarray(X_coords_matched, dtype=np.float64)
    n = min(len(X_coords_all), len(pred_enc))
    X_coords_all = X_coords_all[:n]
    pred_enc = pred_enc[:n]

    valid = (pred_enc >= 0) & (pred_enc < n_cls)
    if not np.all(valid):
        n_bad = int((~valid).sum())
        print(f"  ⚠ Dropping {n_bad} out-of-range tier predictions for spatial plot.")
    X_coords_all = X_coords_all[valid]
    pred_enc = pred_enc[valid]

    pred_names = np.array(
        [names[int(i)] if 0 <= int(i) < n_cls else f"Unknown_{int(i)}" for i in pred_enc]
    )
    pred_df = pd.DataFrame(
        {
            "X_pix_HE": X_coords_all[:, 0],
            "Y_pix_HE": X_coords_all[:, 1],
            "celltype": pred_names,
        }
    )

    if y_tier_f is not None:
        y_true = np.asarray(y_tier_f)[: len(pred_enc)]
        y_true = y_true[valid]
        if y_true.dtype.kind in ["U", "S", "O"] or (
            len(y_true) > 0 and isinstance(y_true.flat[0], (str, np.str_))
        ):
            pred_df["true_celltype"] = y_true
        else:
            y_num = y_true.astype(np.int64)
            pred_df["true_celltype"] = np.array(
                [
                    names[int(i)] if 0 <= int(i) < n_cls else f"Unknown_{int(i)}"
                    for i in y_num
                ]
            )

    plot_celltype_spatial_distribution(
        pred_df,
        x_col="X_pix_HE",
        y_col="Y_pix_HE",
        celltype_col="celltype",
        figsize=fig_size,
        alpha=0.6,
        s=spatial_point_size,
        format="jpg",
        save_path=save_path_pred,
        spatial_color_scheme=spatial_color_scheme,
        color_overrides=color_overrides,
        title=title_pred or "Predicted spatial distribution",
        show=show,
    )
    if save_path_pred:
        print(f"  Saved predicted spatial plot: {save_path_pred}")

    if "true_celltype" in pred_df.columns:
        plot_celltype_spatial_distribution(
            pred_df,
            x_col="X_pix_HE",
            y_col="Y_pix_HE",
            celltype_col="true_celltype",
            figsize=fig_size,
            alpha=0.6,
            s=spatial_point_size,
            format="jpg",
            save_path=save_path_true,
            spatial_color_scheme=spatial_color_scheme,
            color_overrides=color_overrides,
            title=title_true or "Ground-truth spatial distribution",
            show=show,
        )
        if save_path_true:
            print(f"  Saved ground-truth spatial plot: {save_path_true}")


##############################################################
# 2026.02.26 LLY Plot per-class F1-score performance
##############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


def _safe_count_model_parameters(model):
    """
    Count trainable-parameter tensors only for real ``torch.nn.Module`` instances.
    Notebooks sometimes shadow ``model`` with a Python module; avoid crashing there.
    """
    if model is None:
        return None
    try:
        import torch.nn as nn
    except ImportError:
        return None
    if not isinstance(model, nn.Module):
        return None
    try:
        return sum(p.numel() for p in model.parameters())
    except (TypeError, RuntimeError):
        return None


def plot_per_class_f1(
    test_labels,
    test_preds,
    class_names,
    model=None,
    device=None,
    train_dataset=None,
    val_dataset=None,
    input_dim=None,
    best_epoch=None,
    test_acc=None,
    test_macro_f1=None,
    test_weighted_f1=None,
    model_name="MLP",
    save_path=None,
    y_sort_by="support_desc",
):
    """
    Compute per-class metrics and visualize F1-score performance.

    Args:
        test_labels: array-like
        test_preds: array-like
        class_names: full class name list (index-aligned)
        model: torch model (optional, for param counting)
        device: torch device (optional)
        train_dataset, val_dataset: dataset objects (optional)
        input_dim: feature dimension (optional)
        best_epoch: int (optional)
        test_acc, test_macro_f1, test_weighted_f1: float (optional)
        model_name: str
        save_path: str or None
        y_sort_by: str, how to order classes on the y-axis for ``barh``:
            ``"support_desc"`` (default) — by true-label cell count per class, descending
            (most cells at top); ``"f1_asc"`` — by F1 ascending (legacy behavior).
    """

    print("Visualizing per-class performance...")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    test_labels, test_preds = _filter_indices_for_class_names(test_labels, test_preds, class_names)
    if len(test_labels) == 0:
        print("  ⚠ No valid label/pred pairs after filtering; skip per-class F1 plot.")
        return None

    # Only keep classes that actually appear
    unique_labels = np.unique(np.concatenate([test_labels, test_preds]))
    actual_class_names = [class_names[i] for i in unique_labels]

    # Compute per-class metrics
    per_class_metrics = precision_recall_fscore_support(
        test_labels,
        test_preds,
        labels=unique_labels,
        zero_division=0
    )

    metrics_df = pd.DataFrame({
        'Precision': per_class_metrics[0],
        'Recall': per_class_metrics[1],
        'F1-Score': per_class_metrics[2],
        'Support': per_class_metrics[3]
    }, index=actual_class_names)

    print(f"  ✓ Computed metrics for {len(actual_class_names)} classes")

    # Y-axis order: barh puts row 0 at bottom, last row at top.
    # support_desc: sort Support ascending → smallest at bottom, largest at top (descending top-to-bottom).
    if y_sort_by == "support_desc":
        metrics_df_sorted = metrics_df.sort_values("Support", ascending=True)
    elif y_sort_by == "f1_asc":
        metrics_df_sorted = metrics_df.sort_values("F1-Score", ascending=True)
    else:
        raise ValueError("y_sort_by must be 'support_desc' or 'f1_asc'")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = np.arange(len(metrics_df_sorted))
    colors = plt.cm.viridis(
        metrics_df_sorted['F1-Score'] /
        max(metrics_df_sorted['F1-Score'].max(), 1e-8)
    )

    bars = ax.barh(x_pos, metrics_df_sorted['F1-Score'], color=colors)

    ax.set_yticks(x_pos)
    ax.set_yticklabels(metrics_df_sorted.index, fontsize=9)
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title(f'Per-Class F1-Score Performance ({model_name})', fontsize=14)
    ax.set_xlim([0, 1.0])
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (idx, row) in enumerate(metrics_df_sorted.iterrows()):
        ax.text(row['F1-Score'] + 0.01, i,
                f"{row['F1-Score']:.3f}",
                va='center',
                fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    # -----------------------
    # Summary
    # -----------------------
    print("\nModel Summary:")

    if device is not None:
        print(f"  Device: {device}")

    if train_dataset is not None:
        print(f"  Training samples: {len(train_dataset)}")

    if val_dataset is not None:
        print(f"  Test samples: {len(val_dataset)}")

    print(f"  Total classes (before filtering): {len(class_names)}")
    print(f"  Actual classes in test set: {len(unique_labels)}")

    if input_dim is not None:
        print(f"  Feature dimension: {input_dim}")

    n_params = _safe_count_model_parameters(model)
    if n_params is not None:
        print(f"  Model parameters: {n_params:,}")

    if best_epoch is not None:
        print(f"  Best epoch: {best_epoch}")

    if test_acc is not None:
        print(f"  Test Accuracy: {test_acc:.4f}")

    if test_macro_f1 is not None:
        print(f"  Test Macro F1: {test_macro_f1:.4f}")

    if test_weighted_f1 is not None:
        print(f"  Test Weighted F1: {test_weighted_f1:.4f}")

    return metrics_df


##############################################################
# 2026.03.24 LLY Plot per-class Accuracy performance
##############################################################
def plot_per_class_accuracy(
    test_labels,
    test_preds,
    class_names,
    model=None,
    device=None,
    train_dataset=None,
    val_dataset=None,
    input_dim=None,
    best_epoch=None,
    test_acc=None,
    test_macro_f1=None,
    test_weighted_f1=None,
    model_name="MLP",
    save_path=None,
    y_sort_by="support_desc",
):
    """
    Compute per-class accuracy (class-wise recall) and visualize performance.

    Signature intentionally mirrors ``plot_per_class_f1`` for drop-in usage.

    y_sort_by:
        ``support_desc`` (default) — y-axis by true-label Support, descending top-to-bottom;
        ``acc_asc`` — legacy: by Accuracy ascending.
    """
    print("Visualizing per-class accuracy...")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    test_labels, test_preds = _filter_indices_for_class_names(test_labels, test_preds, class_names)
    if len(test_labels) == 0:
        print("  ⚠ No valid label/pred pairs after filtering; skip per-class accuracy plot.")
        return None

    unique_labels = np.unique(np.concatenate([test_labels, test_preds]))
    actual_class_names = [class_names[i] for i in unique_labels]

    per_class_metrics = precision_recall_fscore_support(
        test_labels,
        test_preds,
        labels=unique_labels,
        zero_division=0
    )

    # Per-class accuracy for single-label multi-class == recall per class.
    metrics_df = pd.DataFrame({
        'Precision': per_class_metrics[0],
        'Accuracy': per_class_metrics[1],
        'F1-Score': per_class_metrics[2],
        'Support': per_class_metrics[3]
    }, index=actual_class_names)

    print(f"  ✓ Computed metrics for {len(actual_class_names)} classes")

    if y_sort_by == "support_desc":
        metrics_df_sorted = metrics_df.sort_values("Support", ascending=True)
    elif y_sort_by == "acc_asc":
        metrics_df_sorted = metrics_df.sort_values("Accuracy", ascending=True)
    else:
        raise ValueError("y_sort_by must be 'support_desc' or 'acc_asc'")

    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = np.arange(len(metrics_df_sorted))
    colors = plt.cm.viridis(
        metrics_df_sorted['Accuracy'] /
        max(metrics_df_sorted['Accuracy'].max(), 1e-8)
    )

    ax.barh(x_pos, metrics_df_sorted['Accuracy'], color=colors)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(metrics_df_sorted.index, fontsize=9)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title(f'Per-Class Accuracy Performance ({model_name})', fontsize=14)
    ax.set_xlim([0, 1.0])
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (_, row) in enumerate(metrics_df_sorted.iterrows()):
        ax.text(
            row['Accuracy'] + 0.01,
            i,
            f"{row['Accuracy']:.3f}",
            va='center',
            fontsize=8,
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    print("\nModel Summary:")
    if device is not None:
        print(f"  Device: {device}")
    if train_dataset is not None:
        print(f"  Training samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"  Test samples: {len(val_dataset)}")
    print(f"  Total classes (before filtering): {len(class_names)}")
    print(f"  Actual classes in test set: {len(unique_labels)}")
    if input_dim is not None:
        print(f"  Feature dimension: {input_dim}")
    n_params = _safe_count_model_parameters(model)
    if n_params is not None:
        print(f"  Model parameters: {n_params:,}")
    if best_epoch is not None:
        print(f"  Best epoch: {best_epoch}")
    if test_acc is not None:
        print(f"  Test Accuracy: {test_acc:.4f}")
    if test_macro_f1 is not None:
        print(f"  Test Macro F1: {test_macro_f1:.4f}")
    if test_weighted_f1 is not None:
        print(f"  Test Weighted F1: {test_weighted_f1:.4f}")

    return metrics_df


######################################
# 2026.03.24 LLY Plot level1 accuracy from level2 predictions
######################################
def plot_level1_accuracy_from_level2_predictions(
    matched_features_path,
    all_preds,
    class_names,
    class_names_level1,
    y_encoded_f,
    y_level1_encoded_f,
    plot_per_class_accuracy,
    model=None,
    device=None,
    train_dataset=None,
    val_dataset=None,
    input_dim=None,
    best_epoch=None,
    model_name="MLP-StarDist-Level1",
    save_path=None,
    y_sort_by="support_desc",
):
    """
    Build level1 predictions from level2 predictions and plot per-class accuracy.

    This helper is intended for prediction-only workflows (e.g., StarDist) where
    `all_preds` are level2 class indices and true level1 labels are stored in the
    matched npz (`y_level1`).

    y_sort_by: forwarded to ``plot_per_class_accuracy`` (default: sample-count descending on y).
    """
    loaded_data = _load_matched_features(matched_features_path)
    if "y_level1" not in loaded_data:
        print(f"  ⚠ y_level1 not found in {matched_features_path}; skip level1 accuracy plot.")
        return None

    y_true_l1_raw = loaded_data["y_level1"]

    num_l2_classes = len(class_names)
    child_to_parent = np.full(num_l2_classes, -1, dtype=np.int64)
    for l2, l1 in zip(y_encoded_f, y_level1_encoded_f):
        l2_i, l1_i = int(l2), int(l1)
        if child_to_parent[l2_i] == -1:
            child_to_parent[l2_i] = l1_i
        elif child_to_parent[l2_i] != l1_i:
            raise ValueError(
                f"Inconsistent hierarchy mapping for level2 class {l2_i}: "
                f"{child_to_parent[l2_i]} vs {l1_i}"
            )
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing level1 mapping for level2 classes: {missing}")

    preds_l2 = np.asarray(all_preds).astype(np.int64)
    preds_l1 = child_to_parent[preds_l2]

    # Convert true labels to level1 indices.
    if y_true_l1_raw.dtype.kind in ["U", "S", "O"]:
        name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
        y_true_l1 = np.array([name_to_idx.get(str(v), -1) for v in y_true_l1_raw], dtype=np.int64)
    else:
        y_true_l1 = np.asarray(y_true_l1_raw).astype(np.int64)

    n = min(len(y_true_l1), len(preds_l1))
    y_true_l1 = y_true_l1[:n]
    preds_l1 = preds_l1[:n]
    valid_mask = y_true_l1 >= 0
    if not np.any(valid_mask):
        print("  ⚠ No valid aligned level1 labels; skip level1 accuracy plot.")
        return None
    if not np.all(valid_mask):
        n_bad = int((~valid_mask).sum())
        print(f"  ⚠ {n_bad} unknown true level1 labels excluded from accuracy plot.")

    y_true_l1_valid = y_true_l1[valid_mask]
    preds_l1_valid = preds_l1[valid_mask]

    from sklearn.metrics import accuracy_score, f1_score

    acc_l1 = accuracy_score(y_true_l1_valid, preds_l1_valid)
    macro_f1_l1 = f1_score(y_true_l1_valid, preds_l1_valid, average="macro", zero_division=0)
    weighted_f1_l1 = f1_score(y_true_l1_valid, preds_l1_valid, average="weighted", zero_division=0)
    print(
        "  Level1 metrics from level2 preds: "
        f"acc={acc_l1:.4f}, macro_f1={macro_f1_l1:.4f}, weighted_f1={weighted_f1_l1:.4f}"
    )

    metrics_df = plot_per_class_accuracy(
        y_true_l1_valid,
        preds_l1_valid,
        class_names_level1,
        model=model,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=input_dim,
        best_epoch=best_epoch,
        test_acc=acc_l1,
        test_macro_f1=macro_f1_l1,
        test_weighted_f1=weighted_f1_l1,
        model_name=model_name,
        save_path=save_path,
        y_sort_by=y_sort_by,
    )
    return metrics_df


#############################################################
## 2026.05.11 LLY add 
#############################################################
def plot_level1_accuracy_from_level1_head(
    matched_features_path,
    model,
    device,
    scaler,
    class_names_level1,
    plot_per_class_accuracy,
    train_dataset=None,
    val_dataset=None,
    input_dim=None,
    best_epoch=None,
    model_name="MLP-StarDist-Level1-head",
    save_path=None,
    y_sort_by="support_desc",
    x_key="X",
    batch_size=1024,
    neighbor_index=None,
):
    """
    Run the dedicated level1 logits head (``model.forward_heads``) on matched ``X``,
    compare argmax(L1 logits) to true ``y_level1`` in the npz, and draw the same
    per-class accuracy bar chart as ``plot_per_class_accuracy``.

    Returns ``None`` if ``y_level1`` is missing, ``X`` is missing, or the model has no
    ``forward_heads`` (single-head checkpoints).
    """
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, TensorDataset

    if not (hasattr(model, "forward_heads") and callable(getattr(model, "forward_heads"))):
        print("  ⚠ model has no forward_heads; skip level1-head accuracy plot.")
        return None

    loaded_data = _load_matched_features(matched_features_path)
    if x_key not in loaded_data:
        print(f"  ⚠ {x_key!r} not found in {matched_features_path}; skip level1-head accuracy plot.")
        return None
    if "y_level1" not in loaded_data:
        print(f"  ⚠ y_level1 not found in {matched_features_path}; skip level1-head accuracy plot.")
        return None

    X = np.asarray(loaded_data[x_key], dtype=np.float32)
    y_true_l1_raw = loaded_data["y_level1"]
    n_l1 = len(class_names_level1)

    Xs = scaler.transform(X)
    pin = torch.cuda.is_available()
    ds = TensorDataset(
        torch.from_numpy(Xs),
        torch.zeros(len(Xs), dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, pin_memory=pin)

    model.eval()
    use_spatial = bool(getattr(model, "use_spatial_context", False)) and neighbor_index is not None
    X_full_t = nbr_idx_t = None
    if use_spatial:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        nbr_idx_t = torch.as_tensor(np.asarray(neighbor_index), dtype=torch.long, device=dev)
        X_full_t = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    pl = []
    row_start = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            neighbor_x = _spatial_neighbor_x_for_batch(X_full_t, nbr_idx_t, device, row_start, xb)
            row_start += xb.shape[0]
            logits_l1 = _logits_l1_head_from_model(model, xb, neighbor_x=neighbor_x)
            pl.append(torch.argmax(logits_l1.float(), dim=1).cpu().numpy())
    preds_l1_head = np.concatenate(pl, axis=0)

    if y_true_l1_raw.dtype.kind in ["U", "S", "O"]:
        name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
        y_true_l1 = np.array([name_to_idx.get(str(v), -1) for v in y_true_l1_raw], dtype=np.int64)
    else:
        y_true_l1 = np.asarray(y_true_l1_raw).astype(np.int64)

    n = min(len(y_true_l1), len(preds_l1_head))
    y_true_l1 = y_true_l1[:n]
    preds_l1_head = preds_l1_head[:n]

    valid_mask = (y_true_l1 >= 0) & (preds_l1_head >= 0) & (preds_l1_head < n_l1)
    if not np.any(valid_mask):
        print("  ⚠ No valid aligned level1-head predictions; skip level1-head accuracy plot.")
        return None
    if not np.all(valid_mask):
        n_bad = int((~valid_mask).sum())
        print(f"  ⚠ {n_bad} samples excluded (invalid true L1 or out-of-range L1-head preds).")

    y_v = y_true_l1[valid_mask]
    p_v = preds_l1_head[valid_mask]

    acc_l1 = accuracy_score(y_v, p_v)
    macro_f1_l1 = f1_score(y_v, p_v, average="macro", zero_division=0)
    weighted_f1_l1 = f1_score(y_v, p_v, average="weighted", zero_division=0)
    print(
        "  Level1 metrics from L1 head: "
        f"acc={acc_l1:.4f}, macro_f1={macro_f1_l1:.4f}, weighted_f1={weighted_f1_l1:.4f}"
    )

    metrics_df = plot_per_class_accuracy(
        y_v,
        p_v,
        class_names_level1,
        model=model,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=input_dim,
        best_epoch=best_epoch,
        test_acc=acc_l1,
        test_macro_f1=macro_f1_l1,
        test_weighted_f1=weighted_f1_l1,
        model_name=model_name,
        save_path=save_path,
        y_sort_by=y_sort_by,
    )
    return metrics_df


def plot_level1_f1_from_level1_head(
    matched_features_path,
    model,
    device,
    scaler,
    class_names_level1,
    plot_per_class_f1,
    train_dataset=None,
    val_dataset=None,
    input_dim=None,
    best_epoch=None,
    model_name="MLP-StarDist-Level1-head",
    save_path=None,
    y_sort_by="support_desc",
    x_key="X",
    batch_size=1024,
    neighbor_index=None,
):
    """
    Same inference path as ``plot_level1_accuracy_from_level1_head``, but plots
    per-class **F1** via ``plot_per_class_f1``.
    """
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, TensorDataset

    if not (hasattr(model, "forward_heads") and callable(getattr(model, "forward_heads"))):
        print("  ⚠ model has no forward_heads; skip level1-head F1 plot.")
        return None

    loaded_data = _load_matched_features(matched_features_path)
    if x_key not in loaded_data:
        print(f"  ⚠ {x_key!r} not found in {matched_features_path}; skip level1-head F1 plot.")
        return None
    if "y_level1" not in loaded_data:
        print(f"  ⚠ y_level1 not found in {matched_features_path}; skip level1-head F1 plot.")
        return None

    X = np.asarray(loaded_data[x_key], dtype=np.float32)
    y_true_l1_raw = loaded_data["y_level1"]
    n_l1 = len(class_names_level1)

    Xs = scaler.transform(X)
    pin = torch.cuda.is_available()
    ds = TensorDataset(
        torch.from_numpy(Xs),
        torch.zeros(len(Xs), dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, pin_memory=pin)

    model.eval()
    use_spatial = bool(getattr(model, "use_spatial_context", False)) and neighbor_index is not None
    X_full_t = nbr_idx_t = None
    if use_spatial:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        nbr_idx_t = torch.as_tensor(np.asarray(neighbor_index), dtype=torch.long, device=dev)
        X_full_t = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    pl = []
    row_start = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            neighbor_x = _spatial_neighbor_x_for_batch(X_full_t, nbr_idx_t, device, row_start, xb)
            row_start += xb.shape[0]
            logits_l1 = _logits_l1_head_from_model(model, xb, neighbor_x=neighbor_x)
            pl.append(torch.argmax(logits_l1.float(), dim=1).cpu().numpy())
    preds_l1_head = np.concatenate(pl, axis=0)

    if y_true_l1_raw.dtype.kind in ["U", "S", "O"]:
        name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
        y_true_l1 = np.array([name_to_idx.get(str(v), -1) for v in y_true_l1_raw], dtype=np.int64)
    else:
        y_true_l1 = np.asarray(y_true_l1_raw).astype(np.int64)

    n = min(len(y_true_l1), len(preds_l1_head))
    y_true_l1 = y_true_l1[:n]
    preds_l1_head = preds_l1_head[:n]

    valid_mask = (y_true_l1 >= 0) & (preds_l1_head >= 0) & (preds_l1_head < n_l1)
    if not np.any(valid_mask):
        print("  ⚠ No valid aligned level1-head predictions; skip level1-head F1 plot.")
        return None
    if not np.all(valid_mask):
        n_bad = int((~valid_mask).sum())
        print(f"  ⚠ {n_bad} samples excluded (invalid true L1 or out-of-range L1-head preds).")

    y_v = y_true_l1[valid_mask]
    p_v = preds_l1_head[valid_mask]

    acc_l1 = accuracy_score(y_v, p_v)
    macro_f1_l1 = f1_score(y_v, p_v, average="macro", zero_division=0)
    weighted_f1_l1 = f1_score(y_v, p_v, average="weighted", zero_division=0)
    print(
        "  Level1 F1 plot from L1 head: "
        f"acc={acc_l1:.4f}, macro_f1={macro_f1_l1:.4f}, weighted_f1={weighted_f1_l1:.4f}"
    )

    metrics_df = plot_per_class_f1(
        y_v,
        p_v,
        class_names_level1,
        model=model,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=input_dim,
        best_epoch=best_epoch,
        test_acc=acc_l1,
        test_macro_f1=macro_f1_l1,
        test_weighted_f1=weighted_f1_l1,
        model_name=model_name,
        save_path=save_path,
        y_sort_by=y_sort_by,
    )
    return metrics_df


#############################################################
## 2026.05.11 LLY add 
#############################################################
def plot_level1_f1_from_level2_predictions(
    matched_features_path,
    all_preds,
    class_names,
    class_names_level1,
    y_encoded_f,
    y_level1_encoded_f,
    plot_per_class_f1,
    model=None,
    device=None,
    train_dataset=None,
    val_dataset=None,
    input_dim=None,
    best_epoch=None,
    model_name="MLP-StarDist-Level1",
    save_path=None,
    y_sort_by="support_desc",
):
    """
    Same hierarchy mapping as ``plot_level1_accuracy_from_level2_predictions``,
    but draws per-class **F1** via ``plot_per_class_f1`` (Level1 from Level2 preds).
    """
    loaded_data = _load_matched_features(matched_features_path)
    if "y_level1" not in loaded_data:
        print(f"  ⚠ y_level1 not found in {matched_features_path}; skip level1 F1 plot.")
        return None

    y_true_l1_raw = loaded_data["y_level1"]

    num_l2_classes = len(class_names)
    child_to_parent = np.full(num_l2_classes, -1, dtype=np.int64)
    for l2, l1 in zip(y_encoded_f, y_level1_encoded_f):
        l2_i, l1_i = int(l2), int(l1)
        if child_to_parent[l2_i] == -1:
            child_to_parent[l2_i] = l1_i
        elif child_to_parent[l2_i] != l1_i:
            raise ValueError(
                f"Inconsistent hierarchy mapping for level2 class {l2_i}: "
                f"{child_to_parent[l2_i]} vs {l1_i}"
            )
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing level1 mapping for level2 classes: {missing}")

    preds_l2 = np.asarray(all_preds).astype(np.int64)
    preds_l1 = child_to_parent[preds_l2]

    if y_true_l1_raw.dtype.kind in ["U", "S", "O"]:
        name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
        y_true_l1 = np.array([name_to_idx.get(str(v), -1) for v in y_true_l1_raw], dtype=np.int64)
    else:
        y_true_l1 = np.asarray(y_true_l1_raw).astype(np.int64)

    n = min(len(y_true_l1), len(preds_l1))
    y_true_l1 = y_true_l1[:n]
    preds_l1 = preds_l1[:n]
    valid_mask = y_true_l1 >= 0
    if not np.any(valid_mask):
        print("  ⚠ No valid aligned level1 labels; skip level1 F1 plot.")
        return None
    if not np.all(valid_mask):
        n_bad = int((~valid_mask).sum())
        print(f"  ⚠ {n_bad} unknown true level1 labels excluded from F1 plot.")

    y_true_l1_valid = y_true_l1[valid_mask]
    preds_l1_valid = preds_l1[valid_mask]

    from sklearn.metrics import accuracy_score, f1_score

    acc_l1 = accuracy_score(y_true_l1_valid, preds_l1_valid)
    macro_f1_l1 = f1_score(y_true_l1_valid, preds_l1_valid, average="macro", zero_division=0)
    weighted_f1_l1 = f1_score(y_true_l1_valid, preds_l1_valid, average="weighted", zero_division=0)
    print(
        "  Level1 F1 plot from level2 preds (same mapping as accuracy helper): "
        f"acc={acc_l1:.4f}, macro_f1={macro_f1_l1:.4f}, weighted_f1={weighted_f1_l1:.4f}"
    )

    metrics_df = plot_per_class_f1(
        y_true_l1_valid,
        preds_l1_valid,
        class_names_level1,
        model=model,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=input_dim,
        best_epoch=best_epoch,
        test_acc=acc_l1,
        test_macro_f1=macro_f1_l1,
        test_weighted_f1=weighted_f1_l1,
        model_name=model_name,
        save_path=save_path,
        y_sort_by=y_sort_by,
    )
    return metrics_df


######################################
# 2026.04.13 LLY add Sun plot of celltype distribution
######################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from textwrap import wrap

def plot_sunburst(
    df,
    save_path,
    l1_col="celltype_level1",
    l2_col="celltype",
    center_text="ESCC",
    cmap_name="RdYlBu_r",
    scale_factor=1.2,
    outer_radius=0.90,
    outer_width=0.34,
    outside_label_radius=0.98,
    inside_label_radius=0.72,
    inner_radius=0.56,
    inner_width=0.46,
    colorbar_label="Proportion within layer",
    fig_title="CODEX cell-type hierarchy (L1-L2)",
    font_family=None,
    colorbar_vmin=None,
    colorbar_vmax=None,
    dpi=100,
):
    """
    绘制CODEX细胞类型的Sunburst风格分层环形图
    参数
    ----
    df: DataFrame，必须包含l1_col和l2_col列
    save_path: str，保存PDF的完整路径
    l1_col: str, 一级细胞类型列名
    l2_col: str, 二级细胞类型列名
    center_text: str, 圆心文字
    cmap_name: str, 颜色映射
    figsize: tuple, 图尺寸
    outer_radius: float, 外圈半径
    outer_width: float, 外圈宽度
    outside_label_radius: float, 外圈标签半径
    inside_label_radius: float, 内圈标签半径
    inner_radius: float, 内圈半径
    inner_width: float, 内圈宽度
    colorbar_label: str, 色标名称
    fig_title: str, 图标题
    font_family: list, 字体优先级列表
    colorbar_vmin/vmax: float, 外圈色标范围
    dpi: int, 分辨率
    """
    ## Font fallback list to avoid findfont warnings
    if font_family is None:
        font_family = ["DejaVu Sans", "Liberation Sans", "Noto Sans", "Arial", "Helvetica"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": font_family,
        "axes.labelsize": 15,
        "axes.titlesize": 18,
    })

    plot_df = df[[l1_col, l2_col]].copy()
    ## Level 1
    l1_counts = (
        plot_df.groupby(l1_col, as_index=False)
        .size()
        .rename(columns={"size": "count_l1"})
        .sort_values(l1_col)
    )
    l1_counts["prop_l1_track"] = l1_counts["count_l1"] / l1_counts["count_l1"].sum()

    ## Level2
    l2_counts = (
        plot_df.groupby([l1_col, l2_col], as_index=False)
        .size()
        .rename(columns={"size": "count_l2"})
    )
    l2_counts = l2_counts.merge(l1_counts[[l1_col, "count_l1"]], on=l1_col, how="left")
    l2_counts["prop_l2_within_l1"] = l2_counts["count_l2"] / l2_counts["count_l1"]
    l2_counts = l2_counts.sort_values([l1_col, l2_col]).reset_index(drop=True)

    sizes_l1 = l1_counts["count_l1"].to_numpy()
    labels_l1 = l1_counts[l1_col].to_list()
    props_l1 = l1_counts["prop_l1_track"].to_numpy()

    sizes_l2 = l2_counts["count_l2"].to_numpy()
    labels_l2_raw = l2_counts[l2_col].astype(str).to_list()
    props_l2 = l2_counts["prop_l2_within_l1"].to_numpy()

    ## 包装标签
    def _wrap_label(txt, width=9):
        if len(txt) <= width:
            return txt
        return "\n".join(wrap(txt, width=width, break_long_words=False, break_on_hyphens=False))

    labels_l2 = [_wrap_label(x, width=9) for x in labels_l2_raw]

    def _font_size_for_label(txt):
        n = len(txt)
        if n >= 18:
            return 7
        if n >= 12:
            return 8
        return 9

    cmap = mpl.colormaps[cmap_name]
    norm_l1 = mpl.colors.Normalize(vmin=props_l1.min(), vmax=props_l1.max())
    _vmin = props_l2.min() if colorbar_vmin is None else colorbar_vmin
    _vmax = props_l2.max() if colorbar_vmax is None else colorbar_vmax
    norm_l2 = mpl.colors.Normalize(vmin=_vmin, vmax=_vmax)

    colors_l1 = [cmap(norm_l1(v)) for v in props_l1]
    colors_l2 = [cmap(norm_l2(v)) for v in props_l2]

    fig, ax = plt.subplots(figsize=(12.5/scale_factor, 11.5/scale_factor), subplot_kw=dict(aspect="equal"), dpi=dpi)

    ## Outer ring: level2
    wedges_l2, _ = ax.pie(
        sizes_l2,
        radius=outer_radius,
        labels=None,
        wedgeprops=dict(width=outer_width, edgecolor="white", linewidth=1),
        colors=colors_l2,
        startangle=90,
    )

    LONG_LABEL_THRESHOLD = 12
    LOW_PROP_THRESHOLD = 0.08
    for wedge, raw, wrapped, prop in zip(wedges_l2, labels_l2_raw, labels_l2, props_l2):
        theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        x_edge, y_edge = np.cos(theta) * outer_radius, np.sin(theta) * outer_radius

        needs_outside = (len(raw) > LONG_LABEL_THRESHOLD) or (prop < LOW_PROP_THRESHOLD)
        if needs_outside:
            r_text = outside_label_radius
            x_text, y_text = np.cos(theta) * r_text, np.sin(theta) * r_text
            ha = "left" if x_text >= 0 else "right"
            ax.annotate(
                wrapped,
                xy=(x_edge, y_edge),
                xytext=(x_text, y_text),
                ha=ha,
                va="center",
                fontsize=_font_size_for_label(raw),
                color="#4a4a4a",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666", shrinkA=0, shrinkB=1),
            )
        else:
            r_text = inside_label_radius
            x_text, y_text = np.cos(theta) * r_text, np.sin(theta) * r_text
            ax.text(
                x_text,
                y_text,
                wrapped,
                ha="center",
                va="center",
                rotation=np.rad2deg(theta),
                rotation_mode="anchor",
                fontsize=_font_size_for_label(raw),
                color="#4a4a4a",
            )

    ## Inner ring
    ax.pie(
        sizes_l1,
        radius=inner_radius,
        labels=labels_l1,
        labeldistance=inner_radius - 0.10,
        rotatelabels=True,
        textprops={"fontsize": 14, "color": "#4a4a4a"},
        wedgeprops=dict(width=inner_width, edgecolor="white", linewidth=1),
        colors=colors_l1,
        startangle=90,
    )

    ## Center text
    ax.text(0, 0, center_text, ha="center", va="center", fontsize=14, color="#4a4a4a")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_l2)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label(colorbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    ax.set_title(fig_title, fontsize=16, pad=16)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()




######################################
# 2026.04.13 LLY add histogram + right-side zoom-in panel for heavy-tail distribution
######################################
import matplotlib.pyplot as plt

def plot_cell_area_histogram_zoom(
    df,
    col_name='Cell: Area µm^2',
    bins=120,
    figsize=(9, 3),
    zoom_xlim=(400, 1400),
    save_path=None,
):
    """Plot full histogram and a right-side zoom-in panel on x in [400, 1400]."""
    cell_area = df[col_name].dropna()

    fig, (ax_main, ax_zoom) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.2, 1.2]}
    )

    # Main (full-range) histogram
    ax_main.hist(cell_area, bins=bins, edgecolor='black', alpha=0.75, color='steelblue')
    ax_main.set_xlabel('Cell Area (µm²)')
    ax_main.set_ylabel('Frequency')
    ax_main.set_title('Cell Area Distribution (Full Range)')
    ax_main.grid(True, alpha=0.25, linestyle='--', axis='y')

    # Zoom-in histogram: use only data within zoom window and reset y-scale to zoom counts
    zoom_data = cell_area[(cell_area >= zoom_xlim[0]) & (cell_area <= zoom_xlim[1])]
    n_zoom, _, _ = ax_zoom.hist(
        zoom_data,
        bins=bins,
        range=zoom_xlim,
        edgecolor='black',
        alpha=0.75,
        color='darkorange',
    )
    ax_zoom.set_xlim(*zoom_xlim)
    # Make right y-axis reflect zoomed distribution (avoid inheriting visually huge scale)
    y_max = float(n_zoom.max()) if len(n_zoom) > 0 else 0.0
    ax_zoom.set_ylim(0, max(1.0, y_max * 1.1))
    ax_zoom.set_xlabel('Cell Area (µm²)')
    ax_zoom.set_ylabel('Frequency (zoom)')
    ax_zoom.set_title(f'Zoom-in ({zoom_xlim[0]}-{zoom_xlim[1]})')
    ax_zoom.grid(True, alpha=0.25, linestyle='--', axis='y')

    # Optional visual guides on full-range panel
    ax_main.axvline(zoom_xlim[0], color='gray', linestyle='--', linewidth=1)
    ax_main.axvline(zoom_xlim[1], color='gray', linestyle='--', linewidth=1)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, transparent=True, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


######################################
# 2026.04.13 LLY add correlation plot of PCF and HE cell counts across all tumors
######################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import matplotlib.ticker as mticker

def plot_WSI_counts_cor(
    merged,
    fig_size=(5.2, 4.6),
    transparent=True,
    format='pdf',
    dpi=300,
    save_path=None,
    dataset: str = "CODEX_escc",
):
    """
    Plot and compute correlation between PCF and HE cell counts across all tumors.

    Parameters:
        merged (pd.DataFrame): DataFrame containing columns for tumor IDs, PCF counts, and HE counts.
        transparent (bool): If saving, whether background is transparent.
        format (str): File format for saving figure.
        dpi (int): DPI for saved figure.
        save_path (str or None): If given, save the figure to this path.
        dataset (str): ``CODEX_escc`` (default) or ``Xenium_lung`` — controls axis labels.

    Returns:
        None. Prints correlation stats, shows and optionally saves the plot.
    """
    # Try to detect the relevant columns flexibly
    id_col_candidates = ["TumorID", "tumorid", "roi", "ROI"]
    pcf_col_candidates = ["celltype_count", "pcf_cell_count", "PCF_cell_count", "pcf_count"]
    he_col_candidates = ["cell_count", "he_cell_count", "HE_cell_count", "he_count"]

    def _pick_col(df, candidates, role):
        """Pick the first available column from candidates."""
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"Cannot find {role} column. Tried: {candidates}. Existing: {list(df.columns)}")

    # Determine column names
    id_col = _pick_col(merged, id_col_candidates, "TumorID")
    pcf_col = _pick_col(merged, pcf_col_candidates, "PCF count")
    he_col = _pick_col(merged, he_col_candidates, "HE count")

    # Select and clean data
    corr_df = merged[[id_col, pcf_col, he_col]].copy()
    corr_df = corr_df.dropna(subset=[pcf_col, he_col])

    # Aggregate by tumor ID if necessary
    corr_df = corr_df.groupby(id_col, as_index=False)[[pcf_col, he_col]].sum()

    x = corr_df[pcf_col].to_numpy(dtype=float)
    y = corr_df[he_col].to_numpy(dtype=float)

    # Need at least three tumors for correlation
    if len(corr_df) < 3:
        raise ValueError(f"Need >=3 tumors for correlation, got {len(corr_df)}.")

    # Compute correlations
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    print(f"Tumor count used: {len(corr_df)}")
    print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.3e}")
    print(f"Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.3e}")

    ## Display the sorted DataFrame
    # display(corr_df.sort_values(id_col).reset_index(drop=True))

    # Plotting: assign a distinct color for each TumorID
    fig, ax = plt.subplots(figsize=fig_size)
    tumor_ids = corr_df[id_col].astype(str).tolist()
    # Fixed high-contrast 28-color palette for TumorID points.
    # Colors are selected to be visually distinguishable and stable across runs.
    palette_28 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#e377c2",
        "#bcbd22", "#d62728", "#7f7f7f", "#393b79", "#637939", "#8c6d31", "#843c39",
        "#7b4173", "#3182bd", "#31a354", "#756bb1", "#636363", "#e6550d", "#969696",
        "#6baed6", "#74c476", "#9e9ac8", "#fd8d3c", "#fdae6b", "#9ecae1", "#c7e9c0",
    ]
    color_list = [palette_28[i % len(palette_28)] for i in range(len(tumor_ids))]

    for i, tid in enumerate(tumor_ids):
        ax.scatter(
            x[i],
            y[i],
            s=120,
            alpha=0.95,
            color=color_list[i],
            label=tid,
            zorder=3,
        )

    # Add y=x reference line (as in the sample figure)
    max_val = max(float(np.max(x)), float(np.max(y)))
    line_max = max_val * 1.1
    ax.plot(
        [0, line_max],
        [0, line_max],
        linestyle="--",
        linewidth=1.4,
        color="red",
        alpha=0.85,
        dashes=(4, 4),
        zorder=2,
    )

    ax.set_xlim(0, line_max)
    ax.set_ylim(0, line_max)
    if dataset == "Xenium_lung":
        ax.set_xlabel("GT HE_annotation count")
        ax.set_ylabel("StarDist detection count")
    else:
        ax.set_xlabel("PCF cell count")
        ax.set_ylabel("HE cell count")
    ax.set_title("PCF vs HE counts across tumors")
    ax.legend(
        title="TumorID",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=2,
        fontsize=9,
        title_fontsize=9,
        handlelength=1.0,
        handletextpad=0.4,
        borderaxespad=0.5,
        frameon=False,
    )

    ax.grid(alpha=0.25, linestyle="--")
    
    ## Add Pearson r value to the plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pos = xlim[0] + 0.97 * (xlim[1] - xlim[0])
    y_pos = ylim[0] + 0.03 * (ylim[1] - ylim[0])
    ax.text(
        x_pos,
        y_pos,
        f"Pearson r={pearson_r:.3f}",
        ha="right",
        va="bottom",
        fontsize=11,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
    )

    plt.tight_layout(rect=(0, 0, 0.80, 1))

    ## Adjust the tick label format
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(9)
    ax.yaxis.get_offset_text().set_fontsize(9)

    if save_path is not None:
        plt.savefig(save_path, transparent=transparent, format=format, dpi=dpi, bbox_inches='tight')
    plt.show()



######################################
# 2026.05.11 LLY Multiclass ROC (OvR) for level2 / level1-from-L2 probs
######################################
def mlp_collect_softmax_probs(
    model,
    X_scaled_np,
    device,
    batch_size=2048,
    use_cuda_autocast=None,
    neighbor_index=None,
):
    """
    Run the MLP on all rows of ``X_scaled_np`` and return softmax probabilities (N, C).

    ``use_cuda_autocast``: if None, match ``torch.cuda.is_available()`` (same idea as
    ``evaluate_and_plot_on_all_data`` prediction path).

    When the model was trained with ``use_spatial_context=True``, pass ``neighbor_index``
    (kNN index over the same row order as ``X_scaled_np``).
    """
    import numpy as np
    import torch

    from base import gather_neighbor_embeddings

    if use_cuda_autocast is None:
        use_cuda_autocast = torch.cuda.is_available()

    use_spatial = bool(getattr(model, "use_spatial_context", False)) and neighbor_index is not None

    model.eval()
    Xs = np.asarray(X_scaled_np, dtype=np.float32)
    n = Xs.shape[0]
    chunks = []
    X_full_t = nbr_idx_t = None
    if use_spatial:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        nbr_idx_t = torch.as_tensor(np.asarray(neighbor_index), dtype=torch.long, device=dev)
        X_full_t = torch.as_tensor(Xs, dtype=torch.float32, device=dev)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(Xs[start:end]).to(device, non_blocking=True)
            neighbor_x = None
            if use_spatial:
                global_idx = torch.arange(start, end, dtype=torch.long, device=device)
                neighbor_x = gather_neighbor_embeddings(X_full_t, nbr_idx_t, global_idx, device=device)
            if use_cuda_autocast:
                with torch.amp.autocast("cuda"):
                    logits = model(xb, neighbor_x=neighbor_x)
            else:
                logits = model(xb, neighbor_x=neighbor_x)
            prob = torch.softmax(logits.float(), dim=1)
            chunks.append(prob.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def mlp_collect_five_head_softmax_probs(
    model,
    X_scaled_np,
    device,
    batch_size=2048,
    use_cuda_autocast=None,
    neighbor_index=None,
):
    """
    Batch softmax for all supervised heads.

    Returns
    -------
    dict
        Five-head: ``l2``, ``l1``, ``l12``, ``l3``, ``l4`` each ``(N, C)``.
        Dual/single-head: ``{"l2": (N, C)}`` only.

    When the model was trained with ``use_spatial_context=True``, pass ``neighbor_index``
    (kNN index over the same row order as ``X_scaled_np``).
    """
    import numpy as np
    import torch

    from base import gather_neighbor_embeddings

    if use_cuda_autocast is None:
        use_cuda_autocast = torch.cuda.is_available()

    use_spatial = bool(getattr(model, "use_spatial_context", False)) and neighbor_index is not None

    model.eval()
    Xs = np.asarray(X_scaled_np, dtype=np.float32)
    n = Xs.shape[0]
    five_head = hasattr(model, "level12_head")
    has_dual = hasattr(model, "forward_heads") and callable(getattr(model, "forward_heads"))
    head_names = ("l2", "l1", "l12", "l3", "l4") if five_head else ("l2",)
    chunks = {k: [] for k in head_names}
    X_full_t = nbr_idx_t = None
    if use_spatial:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        nbr_idx_t = torch.as_tensor(np.asarray(neighbor_index), dtype=torch.long, device=dev)
        X_full_t = torch.as_tensor(Xs, dtype=torch.float32, device=dev)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(Xs[start:end]).to(device, non_blocking=True)
            neighbor_x = None
            if use_spatial:
                global_idx = torch.arange(start, end, dtype=torch.long, device=device)
                neighbor_x = gather_neighbor_embeddings(
                    X_full_t, nbr_idx_t, global_idx, device=device
                )
            if use_cuda_autocast:
                with torch.amp.autocast("cuda"):
                    if five_head:
                        logits = model.forward_heads(xb, neighbor_x=neighbor_x)
                    elif has_dual:
                        logits = model.forward_heads(xb, neighbor_x=neighbor_x)[:2]
                    else:
                        logits = (model(xb, neighbor_x=neighbor_x),)
            else:
                if five_head:
                    logits = model.forward_heads(xb, neighbor_x=neighbor_x)
                elif has_dual:
                    logits = model.forward_heads(xb, neighbor_x=neighbor_x)[:2]
                else:
                    logits = (model(xb, neighbor_x=neighbor_x),)

            for name, lg in zip(head_names, logits):
                prob = torch.softmax(lg.float(), dim=1)
                chunks[name].append(prob.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def _aggregate_l2_probs_to_l1(probs_l2, child_to_parent, n_l1_classes):
    """Sum fine-class probabilities into parent level1 buckets."""
    out = np.zeros((probs_l2.shape[0], n_l1_classes), dtype=np.float64)
    for j in range(probs_l2.shape[1]):
        pj = int(child_to_parent[j])
        if 0 <= pj < n_l1_classes:
            out[:, pj] += probs_l2[:, j].astype(np.float64, copy=False)
    return out


def plot_multiclass_roc_curves(
    y_true,
    y_score,
    class_names,
    save_path=None,
    title="Multiclass ROC (one-vs-rest)",
    max_curves=28,
    figsize=(5, 5),
    dpi=300,
    threshold_marker=0.5,
    roc_color_scheme="ncrt",
    class_color_overrides=None,
    ncrt_color_tier="celltype",
):
    """
    One-vs-rest ROC for multiclass cell typing: one curve per class (where defined).

    Parameters
    ----------
    y_true : (N,) int
        Ground-truth class indices in ``0 .. C-1``.
    y_score : (N, C) float
        Predicted class probabilities (rows ~ sum to 1).
    class_names : list[str], length C
    save_path : str or None
        If set, saves this figure (extension controls format, e.g. .pdf / .png).
    threshold_marker : float or None
        If not ``None``, for each drawn OvR curve marks the point whose ``roc_curve``
        threshold is closest to this value (OvR score for the positive class), similar
        to a binary ROC operating point. Use ``None`` to disable markers.
    roc_color_scheme : str
        ``ncrt`` (same colors as NCRT spatial plots via ``ncrt_color_tier``) or
        ``xenium_ct`` / ``xenium_lineage`` / ``xenium_auto``.
    class_color_overrides : dict[str, tuple] optional
        Explicit label -> RGBA; overrides ``roc_color_scheme`` / ``ncrt_color_tier``.
    ncrt_color_tier : str
        When ``roc_color_scheme='ncrt'``, tier hint matching spatial ``celltype_col``
        (``celltype``, ``celltype_level1``, ``celltype_level12``, etc.).
    figsize : (float, float)
        ``(figure_width_inches, roc_square_inches)``. The second value is the side
        length of the **square** ROC panel (height = width in data space, in inches).
        Figure height is derived automatically (room for title + labels). Increase the
        first value when the legend has many entries or ``ncol=2`` (see below).

    Returns
    -------
    dict | None
        ``{"macro_auc": float, "per_class_auc": pd.DataFrame}`` or None if skipped.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float)
    n = min(len(y_true), len(y_score))
    y_true, y_score = y_true[:n], y_score[:n]
    n_classes = y_score.shape[1]
    if len(class_names) != n_classes:
        raise ValueError(f"class_names length {len(class_names)} != y_score columns {n_classes}")

    valid = (y_true >= 0) & (y_true < n_classes)
    if not np.any(valid):
        print("  ⚠ plot_multiclass_roc_curves: no valid labels in range; skip.")
        return None
    y_true = y_true[valid]
    y_score = y_score[valid]

    Y = label_binarize(y_true, classes=np.arange(n_classes))
    per_rows = []
    curves = []
    for k in range(n_classes):
        pos = int(Y[:, k].sum())
        if pos == 0 or pos == len(Y):
            continue
        fpr, tpr, thr = roc_curve(Y[:, k], y_score[:, k], drop_intermediate=True)
        ak = auc(fpr, tpr)
        per_rows.append({"class_idx": k, "class_name": str(class_names[k]), "auc_ovr": ak, "n_pos": pos})
        curves.append((k, fpr, tpr, ak, thr))

    # Macro AUROC = mean of evaluable one-vs-rest class AUCs (handles missing global classes per sample).
    if per_rows:
        macro_auc = float(np.mean([r["auc_ovr"] for r in per_rows]))
    else:
        macro_auc = float("nan")

    curves.sort(key=lambda t: t[3], reverse=True)
    sel = curves[: max(1, int(max_curves))]

    ncol = 2 if len(sel) > 20 else 1
    fig_w, roc_side = float(figsize[0]), float(figsize[1])
    if roc_side <= 0 or fig_w <= 0:
        raise ValueError("figsize entries must be positive (width, roc_square_side).")
    # Widen figure when two-column legend needs horizontal space.
    if ncol == 2 and fig_w < roc_side * 1.55:
        fig_w = roc_side * 1.55
    # Axes box in figure-normalized coords: [left, bottom, sq_w, sq_h] with sq_* = roc_side / fig_*.
    left, bottom = 0.12, 0.13
    _axes_top = 0.90
    _axes_right = 0.98
    # Need fig_w >= roc_side / (_axes_right - left) so left + sq_w <= _axes_right.
    min_fig_w = roc_side / max(1e-6, (_axes_right - left))
    if fig_w < min_fig_w:
        fig_w = min_fig_w
    # Need fig_h >= roc_side / (_axes_top - bottom) so bottom + sq_h <= _axes_top (was broken when
    # fig_h = roc_side * 1.22 => sq_h = 1/1.22 always > 0.90 - bottom).
    fig_h = roc_side / max(1e-6, (_axes_top - bottom))

    sq_w = roc_side / fig_w
    sq_h = roc_side / fig_h
    if left + sq_w > _axes_right + 1e-6 or bottom + sq_h > _axes_top + 1e-6:
        raise ValueError(
            f"ROC layout failed (fig_w={fig_w}, fig_h={fig_h}, roc_side={roc_side}); "
            "increase figsize[0] or report this as a bug."
        )

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left, bottom, sq_w, sq_h])

    class_colors = _roc_colors_for_class_names(
        class_names,
        roc_color_scheme=roc_color_scheme,
        class_color_overrides=class_color_overrides,
        ncrt_color_tier=ncrt_color_tier,
    )
    if roc_color_scheme and str(roc_color_scheme).lower() == "ncrt":
        print(f"  ROC colors: ncrt / {ncrt_color_tier} (aligned with spatial palette).")
    elif roc_color_scheme and str(roc_color_scheme).lower() not in ("ncrt",):
        print(f"  ROC colors: {roc_color_scheme} (aligned with Xenium spatial palette).")

    for i, curve in enumerate(sel):
        k, fpr, tpr, ak, thr = curve
        col = class_colors[k]
        ax.plot(
            fpr,
            tpr,
            color=col,
            lw=1.1,
            label=f"{class_names[k]} ({ak:.3f})",
        )
        if threshold_marker is not None and thr is not None and len(thr):
            thr_arr = np.asarray(thr, dtype=float)
            finite = np.isfinite(thr_arr)
            if np.any(finite):
                dist = np.abs(thr_arr - float(threshold_marker))
                dist = np.where(finite, dist, np.inf)
                ji = int(np.argmin(dist))
                ax.plot(
                    fpr[ji],
                    tpr[ji],
                    "o",
                    color=col,
                    ms=5.5,
                    mew=0.65,
                    mec="white",
                    zorder=5,
                    clip_on=False,
                )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.45)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_box_aspect(1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(
        f"{title}\nmacro AUROC = {macro_auc:.4f}  |  top {len(sel)} / {len(curves)} classes",
        fontsize=9,
        pad=6,
    )
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        frameon=False,
        ncol=ncol,
    )
    ax.grid(True, alpha=0.25)
    # Manual margins: avoid tight_layout fighting a physically square ROC axes.
    fig.subplots_adjust(left=left - 0.02, bottom=bottom - 0.03, right=0.995, top=0.94)
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  ✓ Saved ROC figure: {save_path}")
    plt.show()
    plt.close(fig)

    auc_df = pd.DataFrame(per_rows).sort_values("auc_ovr", ascending=False) if per_rows else None
    return {"macro_auc": macro_auc, "per_class_auc": auc_df}


def plot_level1_roc_from_level2_scores(
    matched_features_path,
    y_score_l2,
    class_names,
    class_names_level1,
    y_encoded_f,
    y_level1_encoded_f,
    figsize=(5, 5),
    save_path=None,
    title="Level1 ROC from level2 probabilities",
    max_curves=16,
    roc_color_scheme="xenium_lineage",
    class_color_overrides=None,
    ncrt_color_tier="celltype_level1",
    y_level1_f=None,
):
    """
    Aggregate level2 softmax rows into level1 probabilities (sum over children),
    then plot multiclass OvR ROC vs true ``y_level1`` in the matched npz.

    Mapping ``child_to_parent`` is inferred from ``(y_encoded_f, y_level1_encoded_f)``
    the same way as ``plot_level1_accuracy_from_level2_predictions``.
    """
    import numpy as np

    if y_level1_f is not None:
        y_true_l1_raw = np.asarray(y_level1_f)
    else:
        loaded_data = _load_matched_features(matched_features_path)
        if "y_level1" not in loaded_data:
            print(f"  ⚠ y_level1 not found in {matched_features_path}; skip level1 ROC.")
            return None
        y_true_l1_raw = loaded_data["y_level1"]
    num_l2_classes = len(class_names)
    child_to_parent = np.full(num_l2_classes, -1, dtype=np.int64)
    for l2, l1 in zip(y_encoded_f, y_level1_encoded_f):
        l2_i, l1_i = int(l2), int(l1)
        if child_to_parent[l2_i] == -1:
            child_to_parent[l2_i] = l1_i
        elif child_to_parent[l2_i] != l1_i:
            raise ValueError(
                f"Inconsistent hierarchy mapping for level2 class {l2_i}: "
                f"{child_to_parent[l2_i]} vs {l1_i}"
            )
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing level1 mapping for level2 classes: {missing}")

    probs_l2 = np.asarray(y_score_l2, dtype=float)
    n_l1 = len(class_names_level1)
    probs_l1 = _aggregate_l2_probs_to_l1(probs_l2, child_to_parent, n_l1)

    if y_true_l1_raw.dtype.kind in ["U", "S", "O"]:
        name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
        y_true_l1 = np.array([name_to_idx.get(str(v), -1) for v in y_true_l1_raw], dtype=np.int64)
    else:
        y_true_l1 = np.asarray(y_true_l1_raw).astype(np.int64)

    n = min(len(y_true_l1), probs_l1.shape[0])
    y_true_l1 = y_true_l1[:n]
    probs_l1 = probs_l1[:n]
    valid_mask = (y_true_l1 >= 0) & (y_true_l1 < n_l1)
    if not np.any(valid_mask):
        print("  ⚠ No valid level1 labels for ROC; skip.")
        return None

    return plot_multiclass_roc_curves(
        y_true_l1[valid_mask],
        probs_l1[valid_mask],
        class_names_level1,
        save_path=save_path,
        title=title,
        max_curves=max_curves,
        figsize=figsize,
        roc_color_scheme=roc_color_scheme,
        class_color_overrides=class_color_overrides,
        ncrt_color_tier=ncrt_color_tier,
    )

#############################################################
## 2026.05.11 LLY add ROC from level1 head (softmax)
#############################################################
def plot_level1_roc_from_level1_head(
    matched_features_path,
    model,
    device,
    scaler,
    class_names_level1,
    figsize=(5, 5),
    save_path=None,
    title="Level1 ROC from level1 head (softmax)",
    max_curves=16,
    x_key="X",
    batch_size=1024,
    n_rows_max=None,
    roc_color_scheme="xenium_lineage",
    class_color_overrides=None,
    ncrt_color_tier="celltype_level1",
    neighbor_index=None,
    X_f=None,
    y_level1_f=None,
):
    """
    Run ``forward_heads`` on matched ``X`` (scaled), take softmax of **level1 logits**,
    then multiclass OvR ROC vs true ``y_level1`` in the npz (same label handling as
    ``plot_level1_roc_from_level2_scores``).

    If ``n_rows_max`` is set (e.g. ``m`` from ``min(len(probs_l2), len(all_labels))``),
    only the first that many rows are used so ROCs align with truncated L2 scores.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if not (hasattr(model, "forward_heads") and callable(getattr(model, "forward_heads"))):
        print("  ⚠ model has no forward_heads; skip level1-head ROC.")
        return None

    if X_f is not None and y_level1_f is not None:
        Xs = np.asarray(X_f, dtype=np.float32)
        y_true_l1_raw = np.asarray(y_level1_f)
    else:
        loaded_data = _load_matched_features(matched_features_path)
        if x_key not in loaded_data:
            print(f"  ⚠ {x_key!r} not found in {matched_features_path}; skip level1-head ROC.")
            return None
        if "y_level1" not in loaded_data:
            print(f"  ⚠ y_level1 not found in {matched_features_path}; skip level1-head ROC.")
            return None

        X = np.asarray(loaded_data[x_key], dtype=np.float32)
        y_true_l1_raw = loaded_data["y_level1"]

        if n_rows_max is not None:
            n2 = int(n_rows_max)
            if n2 > 0:
                X = X[:n2]
                y_true_l1_raw = y_true_l1_raw[:n2]

        Xs = scaler.transform(X)

    n_l1 = len(class_names_level1)
    pin = torch.cuda.is_available()
    ds = TensorDataset(
        torch.from_numpy(Xs),
        torch.zeros(len(Xs), dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, pin_memory=pin)

    model.eval()
    use_spatial = bool(getattr(model, "use_spatial_context", False)) and neighbor_index is not None
    X_full_t = nbr_idx_t = None
    if use_spatial:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        nbr_idx_t = torch.as_tensor(np.asarray(neighbor_index), dtype=torch.long, device=dev)
        X_full_t = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    chunks = []
    row_start = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            neighbor_x = _spatial_neighbor_x_for_batch(X_full_t, nbr_idx_t, device, row_start, xb)
            row_start += xb.shape[0]
            logits_l1 = _logits_l1_head_from_model(model, xb, neighbor_x=neighbor_x)
            prob = torch.softmax(logits_l1.float(), dim=1)
            chunks.append(prob.cpu().numpy())
    probs_l1 = np.concatenate(chunks, axis=0)

    if y_true_l1_raw.dtype.kind in ["U", "S", "O"]:
        name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
        y_true_l1 = np.array([name_to_idx.get(str(v), -1) for v in y_true_l1_raw], dtype=np.int64)
    else:
        y_true_l1 = np.asarray(y_true_l1_raw).astype(np.int64)

    n = min(len(y_true_l1), probs_l1.shape[0])
    y_true_l1 = y_true_l1[:n]
    probs_l1 = probs_l1[:n]
    valid_mask = (y_true_l1 >= 0) & (y_true_l1 < n_l1)
    if not np.any(valid_mask):
        print("  ⚠ No valid level1 labels for L1-head ROC; skip.")
        return None

    return plot_multiclass_roc_curves(
        y_true_l1[valid_mask],
        probs_l1[valid_mask],
        class_names_level1,
        save_path=save_path,
        title=title,
        max_curves=max_curves,
        figsize=figsize,
        roc_color_scheme=roc_color_scheme,
        class_color_overrides=class_color_overrides,
        ncrt_color_tier=ncrt_color_tier,
    )



#############################################################
## 2026.06.23 LLY — Xenium lung: GT vs StarDist count plots (DataFrame in, no paths)
#############################################################
from pathlib import Path


def gt_stardist_counts_for_correlation(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Format counts table for ``plot_WSI_counts_cor`` (GT vs StarDist)."""
    return counts_df.rename(
        columns={"sample": "TumorID", "n_gt": "celltype_count", "n_stardist": "cell_count"}
    )



def plot_gt_stardist_cell_counts_bar(
    counts_df: pd.DataFrame,
    *,
    title: str | None = None,
    gt_label: str = "GT HE_annotation",
    stardist_label: str = "StarDist detection",
    figure_size: tuple[float, float] | None = None,
    dpi: int = 300,
    save_path: str | Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """Grouped bar chart of GT vs StarDist cell counts per sample."""
    if counts_df.empty:
        print("[plot_gt_stardist_cell_counts_bar] Empty counts_df, skip plot.")
        return counts_df

    plot_df = counts_df.melt(
        id_vars="sample",
        value_vars=["n_gt", "n_stardist"],
        var_name="source",
        value_name="n_cells",
    )
    label_map = {"n_gt": gt_label, "n_stardist": stardist_label}
    plot_df["source"] = plot_df["source"].map(label_map)

    n = len(counts_df)
    if figure_size is None:
        figure_size = (max(5.0, n * 0.25), max(5.0, 0.25 * n))

    plt.figure(figsize=figure_size)
    sns.barplot(data=plot_df, x="sample", y="n_cells", hue="source")
    plt.xticks(rotation=90)
    plt.ylabel("Number of cells")
    plt.title(title or "GT vs StarDist cell counts")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return counts_df


def plot_gt_stardist_spatial_mapping(
    stardist_coords_df: pd.DataFrame,
    gt_coords_df: pd.DataFrame,
    sample: str,
    *,
    figure_size: tuple[float, float] = (8, 4.5),
    save_path: str | Path | None = None,
    max_scatter_points: int = 500_000,
    show: bool = True,
) -> pd.DataFrame:
    """Side-by-side spatial scatter: StarDist vs GT (full slide, ``{sample}_all``)."""
    roi_id = f"{sample}_all"
    plot_PCF_HE_counts(
        stardist_coords_df,
        gt_coords_df,
        ROI=roi_id,
        figure_size=figure_size,
        save_path=save_path,
        max_scatter_points=max_scatter_points,
    )
    if not show and save_path is None:
        plt.close()
    return pd.DataFrame(
        {
            "TumorID": [roi_id],
            "n_star_cells": [len(stardist_coords_df)],
            "n_pixel_cells": [len(gt_coords_df)],
        }
    )


def plot_gt_stardist_counts_correlation(
    counts_df: pd.DataFrame,
    *,
    dataset: str = "Xenium_lung",    # "CODEX_escc" (default) or "Xenium_lung"
    fig_size: tuple[float, float] = (10, 5),
    save_path: str | Path | None = None,
    min_samples: int = 3,
    show: bool = True,
) -> pd.DataFrame | None:
    """Pearson/Spearman correlation of GT vs StarDist counts across samples."""
    
    if len(counts_df) < min_samples:
        print(
            f"[plot_gt_stardist_counts_correlation] Need >= {min_samples} samples, "
            f"got {len(counts_df)}."
        )
        return None

    merged = gt_stardist_counts_for_correlation(counts_df)

    plot_WSI_counts_cor(
        merged,
        dataset=dataset,
        fig_size=fig_size,
        save_path=save_path,
    )

    if not show:
        plt.close()
    # return merged

#############################################################