# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import json
import os

# Try to import openslide and tifffile
try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False




def get_celltype_coords(celltype, ncrt_anno_df, value, parent_value='tumor1'):
    # filter
    if parent_value is None:
        subset = celltype[celltype['TumorID'].str.contains(value)]
    else:
        # Common case: value is therapy prefix (e.g. "NCRT") and parent_value is ROI (e.g. "tumor1"),
        # while TumorID is like "NCRT_tumor1".
        if 'TumorID' in celltype.columns:
            tumor_id = f"{value}_{parent_value}"
            tumor_ids = celltype['TumorID']
            if (tumor_ids == value).any():
                subset = celltype[tumor_ids == value]
            elif (tumor_ids == tumor_id).any():
                subset = celltype[tumor_ids == tumor_id]
            else:
                # Fallback: try best-effort contains matching
                mask = tumor_ids.astype(str).str.contains(str(value))
                if isinstance(parent_value, str) and parent_value:
                    mask = mask & tumor_ids.astype(str).str.contains(str(parent_value))
                subset = celltype[mask]
        else:
            subset = celltype[celltype['TumorID'] == value]
    print(subset.columns.tolist())
    # add 'Centroid X µm', 'Centroid Y µm'
    subset = merge_coordinates(subset, ncrt_anno_df, parent_value=parent_value)
    return subset


    
# Divide ROIs based on Parent column (tumor1-tumor28)
# Each tumor corresponds to one ROI with its own unique coordinate range
def divide_into_rois_by_parent(df):
    """
    Divide data into 28 ROIs based on Parent column (tumor1-tumor28)
    Each tumor corresponds to one ROI, using actual coordinate boundaries
    """
    df_copy = df.copy()
    
    # Get all unique Parent values (tumor1-tumor28)
    parents = sorted([p for p in df_copy['Parent'].dropna().unique() if 'tumor' in str(p).lower()])
    
    roi_data = {}
    
    for parent in parents:
        # Extract tumor number
        try:
            tumor_num = int(str(parent).replace('tumor', ''))
        except:
            continue
        
        # Filter all cells for this tumor
        mask = df_copy['Parent'] == parent
        tumor_data = df_copy[mask].copy()
        
        if len(tumor_data) == 0:
            continue
        
        # Calculate actual bounding box for this tumor
        x_min = tumor_data['Centroid X µm'].min()
        x_max = tumor_data['Centroid X µm'].max()
        y_min = tumor_data['Centroid Y µm'].min()
        y_max = tumor_data['Centroid Y µm'].max()
        
        # Add some margin (5%)
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        x_start = max(0, x_min - x_margin)
        x_end = x_max + x_margin
        y_start = max(0, y_min - y_margin)
        y_end = y_max + y_margin
        
        roi_data[tumor_num] = {
            'parent': parent,
            'x_range': (x_start, x_end),
            'y_range': (y_start, y_end),
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'data': tumor_data
        }
    
    return roi_data


def read_qptiff_image(file_path, level=0, region=None):
    """
    Read qptiff format image
    
    Parameters:
    -----------
    file_path : str
        Image file path
    level : int
        Pyramid level (0 is highest resolution)
    region : tuple
        (x, y, width, height) region coordinates, if None then read full image
    
    Returns:
    --------
    image : numpy array
        Image array
    """
    if HAS_OPENSLIDE:
        try:
            slide = openslide.OpenSlide(file_path)
            if region is None:
                # Read full image
                level_count = slide.level_count
                if level >= level_count:
                    level = level_count - 1
                dims = slide.level_dimensions[level]
                image = slide.read_region((0, 0), level, dims)
                image = np.array(image)
                # If RGBA, convert to RGB
                if image.shape[2] == 4:
                    image = image[:, :, :3]
            else:
                x, y, w, h = region
                image = slide.read_region((x, y), level, (w, h))
                image = np.array(image)
                if image.shape[2] == 4:
                    image = image[:, :, :3]
            slide.close()
            return image
        except Exception as e:
            print(f"Failed to read with openslide: {e}")
            if HAS_TIFFFILE:
                return read_with_tifffile(file_path, region)
            return None
    elif HAS_TIFFFILE:
        return read_with_tifffile(file_path, region)
    else:
        print("No available image reading library")
        return None

def read_with_tifffile(file_path, region=None):
    """Read image using tifffile"""
    try:
        with tifffile.TiffFile(file_path) as tif:
            if region is None:
                image = tif.asarray()
            else:
                x, y, w, h = region
                image = tif.asarray()[y:y+h, x:x+w]
        return image
    except Exception as e:
        print(f"Failed to read with tifffile: {e}")
        return None


# Visualize cell annotations for a single ROI
def visualize_roi_cells(roi_id, roi_data, he_image_path=None, 
                        pixel_size_um=0.5, 
                        width_PCF=35520, height_PCF=66240, width_HE=47040, height_HE=70560,
                        figsize=(12, 10), save_path=None):
    """
    Visualize cell annotations for a specified ROI
    
    Parameters:
    -----------
    roi_id : int
        ROI number (1-28)
    width_PCF : int
        Width of PCF image
    height_PCF : int
        Height of PCF image
    width_HE : int
        Width of HE image
    height_HE : int
        Height of HE image
    roi_data : dict
        ROI data dictionary
    he_image_path : str
        HE image path, if None then only show scatter plot
    pixel_size_um : float
        Pixel size (micrometers), for coordinate conversion
    figsize : tuple
        Figure size
    save_path : str
        Save path, if None then don't save
    """
    if roi_id not in roi_data:
        print(f"ROI {roi_id} does not exist")
        return
    
    roi_info = roi_data[roi_id]
    data = roi_info['data']
    x_range = roi_info['x_range']
    y_range = roi_info['y_range']
    
    # Convert micrometer coordinates to pixel coordinates
    x_pixels = (data['Centroid X µm'] / pixel_size_um).values
    y_pixels = (data['Centroid Y µm'] / pixel_size_um).values
    
    # Coordinates relative to ROI boundaries
    x_relative = x_pixels - (x_range[0] / pixel_size_um)
    y_relative = y_pixels - (y_range[0] / pixel_size_um)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: If HE image available, show HE image + cell annotations
    ax1 = axes[0]
    if he_image_path and HAS_OPENSLIDE:
        try:
            # Calculate offset between CODEX and HE images
            # CODEX image: 35520 x 66240, HE image: 47040 x 70560
            he_x_offset = (width_HE - width_PCF) / 2  # 5760 pixels
            he_y_offset = (height_HE - height_PCF) / 2  # 2160 pixels
            
            # Read HE image for ROI region
            # Use actual cell coordinate ranges (not the extended range with margin)
            # Add offset to convert CODEX coordinates to HE image coordinates
            x_start_px = int(roi_info['x_min'] / pixel_size_um) + he_x_offset
            y_start_px = int(roi_info['y_min'] / pixel_size_um) + he_y_offset
            width_px = int((roi_info['x_max'] - roi_info['x_min']) / pixel_size_um)
            height_px = int((roi_info['y_max'] - roi_info['y_min']) / pixel_size_um)
            
            # Read appropriate resolution level
            slide = openslide.OpenSlide(he_image_path)
            # Select appropriate resolution level (use lower resolution if region is too large)
            level = 0
            max_dim = max(width_px, height_px)
            for i in range(slide.level_count):
                if max_dim < slide.level_dimensions[i][0] and max_dim < slide.level_dimensions[i][1]:
                    level = i
                    break
            
            # Adjust coordinates to corresponding level
            downsample = slide.level_downsamples[level]
            x_start_level = int(x_start_px / downsample)
            y_start_level = int(y_start_px / downsample)
            width_level = max(1, int(width_px / downsample))
            height_level = max(1, int(height_px / downsample))
            
            # Ensure coordinates are within image bounds
            slide_dims = slide.level_dimensions[level]
            x_start_level = max(0, min(x_start_level, slide_dims[0] - 1))
            y_start_level = max(0, min(y_start_level, slide_dims[1] - 1))
            width_level = min(width_level, slide_dims[0] - x_start_level)
            height_level = min(height_level, slide_dims[1] - y_start_level)
            
            roi_image = slide.read_region((x_start_level, y_start_level), level, (width_level, height_level))
            roi_image = np.array(roi_image)
            if roi_image.shape[2] == 4:
                roi_image = roi_image[:, :, :3]
            
            slide.close()
            
            # Adjust cell coordinates to image coordinates
            # Use actual min coordinates (not the extended range)
            # Note: cell coordinates are already in CODEX space, so we don't add offset here
            # because we're calculating relative to the ROI region we read
            x_cells = (x_pixels - (roi_info['x_min'] / pixel_size_um)) / downsample
            y_cells = (y_pixels - (roi_info['y_min'] / pixel_size_um)) / downsample
            
            # Filter cells that are within the image bounds
            valid_mask = (x_cells >= 0) & (x_cells < width_level) & (y_cells >= 0) & (y_cells < height_level)
            
            ax1.imshow(roi_image, origin='upper')
            if np.any(valid_mask):
                ax1.scatter(x_cells[valid_mask], y_cells[valid_mask], c='red', s=1, alpha=0.3, edgecolors='none')
            ax1.set_title(f'ROI {roi_id} ({roi_info["parent"]}) - HE image + cell annotations\n({len(data)} cells)', fontsize=12)
            ax1.axis('off')
        except Exception as e:
            print(f"Failed to read HE image: {e}")
            # If reading fails, only show scatter plot
            ax1.scatter(x_relative, y_relative, c='red', s=1, alpha=0.3, edgecolors='none')
            ax1.set_title(f'ROI {roi_id} ({roi_info["parent"]}) - Cell locations\n({len(data)} cells)', fontsize=12)
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            ax1.invert_yaxis()
    else:
        # Only show scatter plot
        ax1.scatter(x_relative, y_relative, c='red', s=1, alpha=0.3, edgecolors='none')
        ax1.set_title(f'ROI {roi_id} ({roi_info["parent"]}) - Cell locations\n({len(data)} cells)', fontsize=12)
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.invert_yaxis()
    
    # Right plot: Cell density heatmap
    ax2 = axes[1]
    if len(data) > 0:
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(x_relative, y_relative, bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax2.imshow(hist.T, origin='lower', extent=extent, cmap='hot', aspect='auto')
        ax2.set_title(f'ROI {roi_id} ({roi_info["parent"]}) - Cell density', fontsize=12)
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.invert_yaxis()
        plt.colorbar(im, ax=ax2, label='Cell count')
    else:
        ax2.text(0.5, 0.5, 'No cell data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'ROI {roi_id} ({roi_info["parent"]}) - Cell density', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    return fig

# Visualize all 28 ROIs with HE images and cell annotations (create grid plot)
def visualize_all_rois_with_he(roi_dict, he_image_path=None, pixel_size_um=0.5, 
                                n_rows=7, n_cols=4, figsize=(16, 28), save_path=None):
    """
    Visualize all ROIs with HE images and cell annotations
    Arrange by tumor number (1-28) in 4x7 grid
    """
    if not HAS_OPENSLIDE or he_image_path is None:
        print("HE image path or openslide not available, falling back to density heatmap")
        return visualize_all_rois(roi_dict, he_image_path, pixel_size_um, n_rows, n_cols, figsize, save_path)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Open slide once for all ROIs
    slide = openslide.OpenSlide(he_image_path)
    
    # Calculate offset between CODEX and HE images
    # CODEX image: 35520 x 66240, HE image: 47040 x 70560
    codex_width = 35520
    codex_height = 66240
    he_x_offset = (47040 - codex_width) / 2  # 5760 pixels
    he_y_offset = (70560 - codex_height) / 2  # 2160 pixels
    
    # Arrange by tumor number order
    for idx, roi_id in enumerate(range(1, 29)):  # tumor1 to tumor28
        ax = axes[idx]
        
        if roi_id in roi_dict:
            roi_info = roi_dict[roi_id]
            data = roi_info['data']
            
            # Convert coordinates
            x_pixels = (data['Centroid X µm'] / pixel_size_um).values
            y_pixels = (data['Centroid Y µm'] / pixel_size_um).values
            
            # Use actual cell coordinate ranges
            # Add offset to convert CODEX coordinates to HE image coordinates
            x_start_px = int(roi_info['x_min'] / pixel_size_um) + he_x_offset
            y_start_px = int(roi_info['y_min'] / pixel_size_um) + he_y_offset
            width_px = int((roi_info['x_max'] - roi_info['x_min']) / pixel_size_um)
            height_px = int((roi_info['y_max'] - roi_info['y_min']) / pixel_size_um)
            
            # Select appropriate resolution level
            level = 0
            max_dim = max(width_px, height_px)
            for i in range(slide.level_count):
                if max_dim < slide.level_dimensions[i][0] and max_dim < slide.level_dimensions[i][1]:
                    level = i
                    break
            
            # Adjust coordinates to corresponding level
            downsample = slide.level_downsamples[level]
            x_start_level = int(x_start_px / downsample)
            y_start_level = int(y_start_px / downsample)
            width_level = max(1, int(width_px / downsample))
            height_level = max(1, int(height_px / downsample))
            
            # Ensure coordinates are within image bounds
            slide_dims = slide.level_dimensions[level]
            x_start_level = max(0, min(x_start_level, slide_dims[0] - 1))
            y_start_level = max(0, min(y_start_level, slide_dims[1] - 1))
            width_level = min(width_level, slide_dims[0] - x_start_level)
            height_level = min(height_level, slide_dims[1] - y_start_level)
            
            try:
                roi_image = slide.read_region((x_start_level, y_start_level), level, (width_level, height_level))
                roi_image = np.array(roi_image)
                if roi_image.shape[2] == 4:
                    roi_image = roi_image[:, :, :3]
                
                # Adjust cell coordinates to image coordinates
                # Note: cell coordinates are already in CODEX space, so we don't add offset here
                # because we're calculating relative to the ROI region we read
                x_cells = (x_pixels - (roi_info['x_min'] / pixel_size_um)) / downsample
                y_cells = (y_pixels - (roi_info['y_min'] / pixel_size_um)) / downsample
                
                # Filter cells that are within the image bounds
                valid_mask = (x_cells >= 0) & (x_cells < width_level) & (y_cells >= 0) & (y_cells < height_level)
                
                # ax.imshow(roi_image, origin='upper')
                if len(data) > 0 and np.any(valid_mask):
                    ax.scatter(x_cells[valid_mask], y_cells[valid_mask], c='red', s=0.5, alpha=0.4, edgecolors='none')
                
                ax.set_title(f'ROI {roi_id} ({roi_info["parent"]})\n({len(data)} cells)', fontsize=8)
            except Exception as e:
                print(f"Error reading ROI {roi_id}: {e}")
                ax.text(0.5, 0.5, f'ROI {roi_id}\nError', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8)
            
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, f'ROI {roi_id}\nNo data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    
    slide.close()
    
    plt.suptitle('All 28 ROIs - HE images with cell annotations (arranged by tumor number)', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    return fig

# Visualize all 28 ROIs (create grid plot) - density heatmap version
def visualize_all_rois(roi_dict, he_image_path=None, pixel_size_um=0.5, 
                       n_rows=7, n_cols=4, figsize=(12, 20), save_path=None):
    """
    Visualize cell distribution for all ROIs (density heatmap)
    Arrange by tumor number (1-28) in 4x7 grid
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Arrange by tumor number order
    for idx, roi_id in enumerate(range(1, 29)):  # tumor1 to tumor28
        ax = axes[idx]
        
        if roi_id in roi_dict:
            roi_info = roi_dict[roi_id]
            data = roi_info['data']
            x_range = roi_info['x_range']
            y_range = roi_info['y_range']
            
            # Convert coordinates (relative to ROI boundaries)
            x_pixels = (data['Centroid X µm'] / pixel_size_um).values
            y_pixels = (data['Centroid Y µm'] / pixel_size_um).values
            x_relative = x_pixels - (x_range[0] / pixel_size_um)
            y_relative = y_pixels - (y_range[0] / pixel_size_um)
            
            # Create density heatmap
            if len(data) > 0:
                hist, xedges, yedges = np.histogram2d(x_relative, y_relative, bins=30)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                ax.imshow(hist.T, origin='lower', extent=extent, cmap='hot', aspect='auto')
            
            ax.set_title(f'ROI {roi_id} ({roi_info["parent"]})\n({len(data)} cells)', fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, f'ROI {roi_id}\nNo data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
    
    plt.suptitle('Cell distribution for all 28 ROIs (arranged by tumor number)', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    return fig



def extract_and_save_celltype_by_tumorid(celltype_df, 
    base_path='/home/lingyu/data/Python/Collaborate/esccAI/data'):
    """
    Extract cell type information for each TumorID and save as CSV files.

    Parameters:
    celltype_df: DataFrame, a dataframe containing TumorID and celltype columns
    base_path: str, base path, default is '/home/lingyu/data/Python/Collaborate/esccAI/data'

    Returns:
    dict: contains information about the saved file paths
    """
    import os
    import pandas as pd

    # Create codex_celltype folder
    output_base_dir = os.path.join(base_path, 'codex_celltype')
    os.makedirs(output_base_dir, exist_ok=True)

    # Get all unique TumorIDs
    unique_tumorids = celltype_df['TumorID'].unique()
    print(f"Found {len(unique_tumorids)} unique TumorIDs")

    # For storing information about saved files
    saved_files = {}

    # Process by grouping according to TumorID
    for tumor_id in unique_tumorids:
        # Extract data for this TumorID
        tumor_data = celltype_df[celltype_df['TumorID'] == tumor_id].copy()

        # Extract prefix (e.g. 'NCRT' from 'NCRT_tumor1')
        prefix = tumor_id.split('_')[0]

        # Create prefix folder
        prefix_dir = os.path.join(output_base_dir, prefix)
        os.makedirs(prefix_dir, exist_ok=True)

        # Save as CSV file, filename uses TumorID
        output_file = os.path.join(prefix_dir, f'{tumor_id}.csv')
        tumor_data.to_csv(output_file, index=False)

        # Record saved file information
        if prefix not in saved_files:
            saved_files[prefix] = []
        saved_files[prefix].append({
            'tumor_id': tumor_id,
            'file_path': output_file,
            'num_cells': len(tumor_data)
        })

        print(f"Saved: {output_file} (contains {len(tumor_data)} cells)")

    # Print summary information
    print(f"\nSummary:")
    print(f"Processed {len(unique_tumorids)} TumorIDs in total")
    print(f"Saved in: {output_base_dir}")
    for prefix, files in saved_files.items():
        print(f"  {prefix} folder: {len(files)} files")

    return saved_files

def merge_coordinates(celltype_NCRT_tumor1, ncrt_anno_df1, parent_value='tumor1'):
    """
    Merge Centroid X µm and Centroid Y µm from ncrt_anno_df1 to celltype_NCRT_tumor1 based on CellID.
    Filters ncrt_anno_df1 by 'Parent' column to only include parent_value (default: 'tumor1').
    
    Parameters:
        celltype_NCRT_tumor1 (pd.DataFrame): DataFrame with CellID and cell type info.
        ncrt_anno_df1 (pd.DataFrame): DataFrame with Object ID (to be renamed to CellID), coordinates, and Parent info.
        parent_value (str): Value to filter ncrt_anno_df1['Parent'] (default: 'tumor1').
        
    Returns:
        pd.DataFrame: Merged DataFrame with coordinates added.
    """
    import pandas as pd

    # Rename 'Object ID' to 'CellID'
    if 'Object ID' in ncrt_anno_df1.columns:
        ncrt_anno_df1 = ncrt_anno_df1.rename(columns={'Object ID': 'CellID'})
    
    if parent_value == None:
        print("Dont filtering - parent value: 'Parent' column")
        print("Available columns:", ncrt_anno_df1.columns.tolist()[:10])
        ncrt_anno_df1_tumor1 = ncrt_anno_df1.copy()
    else:       
        print("Before filtering - ncrt_anno_df1 shape:", ncrt_anno_df1.shape)
        if 'Parent' in ncrt_anno_df1.columns:
            print("Unique Parent values:\n", ncrt_anno_df1['Parent'].unique())
            # Filter for specified parent_value
            ncrt_anno_df1_tumor1 = ncrt_anno_df1[ncrt_anno_df1['Parent'] == parent_value].copy()
            print(f"After filtering for '{parent_value}' - ncrt_anno_df1_tumor1 shape:", ncrt_anno_df1_tumor1.shape)
            print(f"Number of cells in {parent_value}: {len(ncrt_anno_df1_tumor1)}")
        else:
            print("Warning: 'Parent' column not found in ncrt_anno_df1")
            print("Available columns:", ncrt_anno_df1.columns.tolist()[:10])
            ncrt_anno_df1_tumor1 = ncrt_anno_df1.copy()

        print("\nShape of celltype_NCRT_tumor1 before merging:", celltype_NCRT_tumor1.shape)
        print("Shape of ncrt_anno_df1_tumor1:", ncrt_anno_df1_tumor1.shape)

    # Check for 'CellID' columns
    if 'CellID' in celltype_NCRT_tumor1.columns and 'CellID' in ncrt_anno_df1_tumor1.columns:
        # Select columns for merging
        coord_cols = ['CellID', 'Centroid X µm', 'Centroid Y µm']
        coord_df = ncrt_anno_df1_tumor1[coord_cols].copy()
        
        # Merge
        merged_df = celltype_NCRT_tumor1.merge(coord_df, on='CellID', how='left')
        
        print("Shape of celltype_NCRT_tumor1 after merging:", merged_df.shape)
        print("\nColumns after merging:", merged_df.columns.tolist())
        # print("\nFirst 5 rows:")
        # print(merged_df.head())
        
        ## Check for missing coordinates
        # missing_coords = merged_df[['Centroid X µm', 'Centroid Y µm']].isnull().sum()
        # print(f"\nNumber of missing coordinates:\n{missing_coords}")
        
        # Check merge success rate
        matched = merged_df[['Centroid X µm', 'Centroid Y µm']].notna().all(axis=1).sum()
        if len(merged_df) == 0:
            print("Successfully matched cells: 0 / 0 (0.00%)")
        else:
            print(f"Successfully matched cells: {matched} / {len(merged_df)} ({matched/len(merged_df)*100:.2f}%)")
        
        return merged_df
    else:
        print("Error: 'CellID' column not found for merging")
        if 'CellID' not in celltype_NCRT_tumor1.columns:
            print("  'CellID' column is missing in celltype_NCRT_tumor1")
        if 'CellID' not in ncrt_anno_df1_tumor1.columns:
            print("  'CellID' column is missing in ncrt_anno_df1_tumor1")
        return celltype_NCRT_tumor1  # return original if merge fails



########################################################
# 2026.02.23 Alignment of PCF and HE images
# The results are from QuPath, by manually aligning
########################################################
def align_matrix(therapy_data):
    if therapy_data == 'NCRT':
        # Original matrix from QuPath (HE -> PCF)
        qupath_matrix_HE2PCF = np.array([
            [0.9837527730111494, -0.00772653508276407, -5461.577260839792],
            [0.00772653508276407, 0.9837527730111494, -2103.182222101196],
        ])
    elif therapy_data == 'NCT':
        qupath_matrix_HE2PCF = np.array([
            [0.9837075824586441, -0.012190568686802504, -3122.133028307581],
            [0.012190568686802504, 0.9837075824586441, -188.23466461087844],
        ])
    elif therapy_data == 'NICT':
        qupath_matrix_HE2PCF = np.array([
            [0.9836722965500848, -0.01476586439716659, 400.5765904268108],
            [0.01476586439716659, 0.9836722965500848, -1455.6620337578101],
        ])
    elif therapy_data == 'SA':
        qupath_matrix_HE2PCF = np.array([
            [0.9836018158963842, -0.01888611968782861, -358.49116395569877],
            [0.01888611968782861, 0.9836018158963842, -601.5145795302255],
        ])
    else:
        raise ValueError(f"Unsupported therapy data: {therapy_data}")
    return qupath_matrix_HE2PCF


## IMPORTANT: In QuPath, NCRT PCF.qptiff is the reference (front) and NCRT HE.qptiff is the target (back)
## The matrix shown in QuPath transforms HE coordinates to PCF coordinates
## Since we need PCF -> HE transformation, we need to INVERT this matrix
def transformation_matrix_properties(therapy_data):

    ## Get the transformation matrix
    qupath_matrix_HE2PCF = align_matrix(therapy_data)

    # Print the original matrix
    print("Original QuPath matrix (HE -> PCF):")
    print(qupath_matrix_HE2PCF)
    print(f"\nMatrix properties (HE -> PCF):")
    scale = qupath_matrix_HE2PCF[0, 0]
    rotation = np.arcsin(qupath_matrix_HE2PCF[1, 0]) * 180 / np.pi
    print(f"  Scale: {scale:.4f}, Rotation: {rotation:.4f}°")
    print(f"  Translation: X={qupath_matrix_HE2PCF[0,2]:.1f}, Y={qupath_matrix_HE2PCF[1,2]:.1f}")

    # Print the after inverting matrix
    PCF2HE_transformation_NCRT = invert_transformation_matrix(qupath_matrix_HE2PCF)
    scale_inv = np.sqrt(PCF2HE_transformation_NCRT[0,0]**2 + PCF2HE_transformation_NCRT[0,1]**2)
    rotation_inv = np.arcsin(PCF2HE_transformation_NCRT[1,0]) * 180 / np.pi
    print("="*60 + "\n")

    print("Inverted matrix (PCF -> HE):")
    print(PCF2HE_transformation_NCRT)
    print(f"\nMatrix properties (PCF -> HE):")
    print(f"  Scale: {scale_inv:.4f}, Rotation: {rotation_inv:.4f}°")
    print(f"  Translation: X={PCF2HE_transformation_NCRT[0,2]:.1f}, Y={PCF2HE_transformation_NCRT[1,2]:.1f}")
    print("="*60)
    return PCF2HE_transformation_NCRT


## Invert to get PCF -> HE transformation
def invert_transformation_matrix(transform_2x3):
    """Invert a 2x3 affine transformation matrix"""
    transform_3x3 = np.array([
        [transform_2x3[0, 0], transform_2x3[0, 1], transform_2x3[0, 2]],
        [transform_2x3[1, 0], transform_2x3[1, 1], transform_2x3[1, 2]],
        [0, 0, 1]
    ])
    inv_transform_3x3 = np.linalg.inv(transform_3x3)
    return inv_transform_3x3[:2, :]


def align_pcf_to_he_coordinates(pcf_coords, transformation_matrix):
    """
    Transform PCF pixel coordinates to HE pixel coordinates using transformation matrix
    
    Parameters:
    -----------
    pcf_coords : array-like, shape (N, 2)
        PCF pixel coordinates [x, y]
    transformation_matrix : array, shape (2, 3)
        Transformation matrix from PCF to HE coordinates
    
    Returns:
    --------
    he_coords : array, shape (N, 2)
        HE pixel coordinates [x, y]
    """
    pcf_coords = np.array(pcf_coords)
    if pcf_coords.ndim == 1:
        pcf_coords = pcf_coords.reshape(1, -1)
    
    # Add homogeneous coordinate
    ones = np.ones((pcf_coords.shape[0], 1))
    pcf_coords_homogeneous = np.hstack([pcf_coords, ones])
    
    # Apply transformation: [x_he, y_he] = [x_pcf, y_pcf, 1] @ T^T
    he_coords = pcf_coords_homogeneous @ transformation_matrix.T
    
    return he_coords


def verify_transformation_alignment(he_path, pcf_path, transformation_matrix, 
                                    test_regions=2, region_size=4000, figsize=(18, 12)):
    """
    Verify the alignment quality of the transformation matrix by testing multiple regions
    
    Parameters:
    -----------
    he_path : str
        Path to HE image
    pcf_path : str
        Path to PCF image
    transformation_matrix : array, shape (2, 3)
        Transformation matrix to verify
    test_regions : int
        Number of test regions per dimension (total = test_regions^2)
    region_size : int
        Size of each test region in pixels
    figsize : tuple
        Figure size
    """
    if not HAS_OPENSLIDE:
        print("OpenSlide not available")
        return None
    
    # Check if files exist
    if not os.path.exists(he_path):
        print(f"HE image not found: {he_path}")
        return None
    if not os.path.exists(pcf_path):
        print(f"PCF image not found: {pcf_path}")
        return None
    
    try:
        # print(f"Opening HE image: {he_path}")
        he_slide = openslide.OpenSlide(he_path)
        he_width, he_height = he_slide.dimensions
        print(f"HE image dimensions: {he_width} x {he_height}")
        
        # print(f"Opening PCF image: {pcf_path}")
        pcf_slide = openslide.OpenSlide(pcf_path)
        pcf_width, pcf_height = pcf_slide.dimensions
        print(f"PCF image dimensions: {pcf_width} x {pcf_height}")
        
        # Generate test points across the PCF image
        test_points_pcf = []
        for i in range(1, test_regions + 1):
            x = int(pcf_width * i / (test_regions + 1))
            for j in range(1, test_regions + 1):
                y = int(pcf_height * j / (test_regions + 1))
                test_points_pcf.append((x, y))
        
        # Transform to HE coordinates
        test_points_he = []
        valid_points = []
        for i, pcf_pt in enumerate(test_points_pcf):
            he_pt = align_pcf_to_he_coordinates([pcf_pt], transformation_matrix)[0]
            test_points_he.append(he_pt)
            # Check if transformed point is within HE image bounds
            if 0 <= he_pt[0] < he_width and 0 <= he_pt[1] < he_height:
                valid_points.append(i)
        
        if len(valid_points) == 0:
            print(f"Warning: No valid transformed points found within HE image bounds")
            print(f"Transformed points range: X=[{min([pt[0] for pt in test_points_he]):.1f}, {max([pt[0] for pt in test_points_he]):.1f}], "
                  f"Y=[{min([pt[1] for pt in test_points_he]):.1f}, {max([pt[1] for pt in test_points_he]):.1f}]")
            print(f"HE image bounds: X=[0, {he_width}], Y=[0, {he_height}]")
        
        print(f"Generated {len(test_points_pcf)} test points, {len(valid_points)} within HE bounds")
        
        # Create visualization - show 2 rows: one for each test region
        # Each row has 3 columns: PCF, HE, Overlay
        n_regions = len(test_points_pcf)
        n_cols = 3  # PCF, HE, Overlay
        n_rows = n_regions
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_regions == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, (pcf_center, he_center) in enumerate(zip(test_points_pcf, test_points_he)):
            if idx >= n_regions:
                break
            
            # Three subplots per region: PCF, HE, Overlay
            ax_pcf_only = axes[idx * 3]
            ax_he_only = axes[idx * 3 + 1]
            ax_overlay = axes[idx * 3 + 2]
            
            try:
                # Define PCF region corners
                half_size = region_size // 2
                pcf_corners = np.array([
                    [pcf_center[0] - half_size, pcf_center[1] - half_size],  # top-left
                    [pcf_center[0] + half_size, pcf_center[1] - half_size],  # top-right
                    [pcf_center[0] + half_size, pcf_center[1] + half_size],  # bottom-right
                    [pcf_center[0] - half_size, pcf_center[1] + half_size],  # bottom-left
                ])
                
                # Clip to image bounds
                pcf_corners[:, 0] = np.clip(pcf_corners[:, 0], 0, pcf_width)
                pcf_corners[:, 1] = np.clip(pcf_corners[:, 1], 0, pcf_height)
                
                # Transform corners to HE coordinates
                he_corners = align_pcf_to_he_coordinates(pcf_corners, transformation_matrix)
                
                # Calculate bounding box in HE coordinates
                x_min_he = max(0, int(he_corners[:, 0].min()))
                y_min_he = max(0, int(he_corners[:, 1].min()))
                x_max_he = min(he_width, int(he_corners[:, 0].max()))
                y_max_he = min(he_height, int(he_corners[:, 1].max()))
                
                # Read PCF region
                x_pcf_start = int(pcf_corners[:, 0].min())
                y_pcf_start = int(pcf_corners[:, 1].min())
                x_pcf_end = int(pcf_corners[:, 0].max())
                y_pcf_end = int(pcf_corners[:, 1].max())
                
                pcf_region_width = x_pcf_end - x_pcf_start
                pcf_region_height = y_pcf_end - y_pcf_start
                
                if pcf_region_width <= 0 or pcf_region_height <= 0:
                    for ax in [ax_pcf_only, ax_he_only, ax_overlay]:
                        ax.text(0.5, 0.5, f'Invalid PCF region', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=8)
                        ax.axis('off')
                    continue
                
                # Read PCF region - use a lower pyramid level for better visibility
                # Try to find an appropriate level
                pcf_level = 0
                max_dim = max(pcf_region_width, pcf_region_height)
                for i in range(pcf_slide.level_count):
                    if max_dim < pcf_slide.level_dimensions[i][0] and max_dim < pcf_slide.level_dimensions[i][1]:
                        pcf_level = i
                        break
                
                pcf_downsample = pcf_slide.level_downsamples[pcf_level]
                x_pcf_level = int(x_pcf_start / pcf_downsample)
                y_pcf_level = int(y_pcf_start / pcf_downsample)
                w_pcf_level = max(1, int(pcf_region_width / pcf_downsample))
                h_pcf_level = max(1, int(pcf_region_height / pcf_downsample))
                
                # Ensure within bounds
                pcf_dims = pcf_slide.level_dimensions[pcf_level]
                x_pcf_level = max(0, min(x_pcf_level, pcf_dims[0] - 1))
                y_pcf_level = max(0, min(y_pcf_level, pcf_dims[1] - 1))
                w_pcf_level = min(w_pcf_level, pcf_dims[0] - x_pcf_level)
                h_pcf_level = min(h_pcf_level, pcf_dims[1] - y_pcf_level)
                
                pcf_region = np.array(pcf_slide.read_region(
                    (x_pcf_level, y_pcf_level), pcf_level,
                    (w_pcf_level, h_pcf_level)
                ))
                if pcf_region.shape[2] == 4:
                    pcf_region = pcf_region[:, :, :3]
                
                # Resize to original size for consistency
                if pcf_level > 0:
                    from PIL import Image
                    pcf_region = Image.fromarray(pcf_region).resize(
                        (pcf_region_width, pcf_region_height),
                        Image.Resampling.LANCZOS
                    )
                    pcf_region = np.array(pcf_region)
                
                # Read HE region (corresponding transformed region)
                # Check if region is valid
                he_region_width = x_max_he - x_min_he
                he_region_height = y_max_he - y_min_he
                
                if he_region_width <= 0 or he_region_height <= 0:
                    for ax in [ax_pcf_only, ax_he_only, ax_overlay]:
                        ax.text(0.5, 0.5, f'Invalid HE region', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=8)
                        ax.axis('off')
                    continue
                
                # Read HE region - also use appropriate level
                he_level = 0
                max_dim_he = max(he_region_width, he_region_height)
                for i in range(he_slide.level_count):
                    if max_dim_he < he_slide.level_dimensions[i][0] and max_dim_he < he_slide.level_dimensions[i][1]:
                        he_level = i
                        break
                
                he_downsample = he_slide.level_downsamples[he_level]
                x_he_level = int(x_min_he / he_downsample)
                y_he_level = int(y_min_he / he_downsample)
                w_he_level = max(1, int(he_region_width / he_downsample))
                h_he_level = max(1, int(he_region_height / he_downsample))
                
                # Ensure within bounds
                he_dims = he_slide.level_dimensions[he_level]
                x_he_level = max(0, min(x_he_level, he_dims[0] - 1))
                y_he_level = max(0, min(y_he_level, he_dims[1] - 1))
                w_he_level = min(w_he_level, he_dims[0] - x_he_level)
                h_he_level = min(h_he_level, he_dims[1] - y_he_level)
                
                he_region = np.array(he_slide.read_region(
                    (x_he_level, y_he_level), he_level,
                    (w_he_level, h_he_level)
                ))
                if he_region.shape[2] == 4:
                    he_region = he_region[:, :, :3]
                
                # Resize to original size for consistency
                if he_level > 0:
                    from PIL import Image
                    he_region = Image.fromarray(he_region).resize(
                        (he_region_width, he_region_height),
                        Image.Resampling.LANCZOS
                    )
                    he_region = np.array(he_region)
                
                # Display original PCF region
                # Normalize and enhance contrast
                if pcf_region.max() > 255 or pcf_region.dtype != np.uint8:
                    pcf_region = np.clip(pcf_region, 0, 255).astype(np.uint8)
                
                # Enhance contrast for better visibility
                pcf_display = pcf_region.copy()
                if pcf_display.max() > 0:
                    # Stretch contrast to full range
                    pcf_min = np.percentile(pcf_display, 2)
                    pcf_max = np.percentile(pcf_display, 98)
                    if pcf_max > pcf_min:
                        pcf_display = np.clip((pcf_display - pcf_min) / (pcf_max - pcf_min) * 255, 0, 255).astype(np.uint8)
                
                ax_pcf_only.imshow(pcf_display)
                ax_pcf_only.set_title(f'R{idx+1} PCF Original\n({pcf_center[0]}, {pcf_center[1]})', fontsize=10)
                ax_pcf_only.axis('off')
                
                # Display HE region
                if he_region.max() > 255 or he_region.dtype != np.uint8:
                    he_region = np.clip(he_region, 0, 255).astype(np.uint8)
                
                # Enhance contrast for HE as well
                he_display = he_region.copy()
                if he_display.max() > 0:
                    he_min = np.percentile(he_display, 2)
                    he_max = np.percentile(he_display, 98)
                    if he_max > he_min:
                        he_display = np.clip((he_display - he_min) / (he_max - he_min) * 255, 0, 255).astype(np.uint8)
                
                ax_he_only.imshow(he_display)
                ax_he_only.set_title(f'R{idx+1} HE Aligned\n({he_center[0]:.0f}, {he_center[1]:.0f})', fontsize=10)
                ax_he_only.axis('off')
                
                # Apply transformation to align PCF with HE
                # For verification, we'll transform the PCF region to match HE region
                # The transformation matrix: HE_coord = T @ PCF_coord
                
                try:
                    import cv2
                    
                    # For cv2.warpAffine, we need the inverse transform
                    # cv2.warpAffine maps: dst(x,y) = src(M11*x + M12*y + M13, M21*x + M22*y + M23)
                    # So we need HE -> PCF mapping
                    
                    # Convert to 3x3 and invert
                    transform_3x3 = np.array([
                        [transformation_matrix[0, 0], transformation_matrix[0, 1], transformation_matrix[0, 2]],
                        [transformation_matrix[1, 0], transformation_matrix[1, 1], transformation_matrix[1, 2]],
                        [0, 0, 1]
                    ])
                    inv_transform_3x3 = np.linalg.inv(transform_3x3)
                    
                    # Get the warp matrix (HE output -> PCF input)
                    warp_matrix = inv_transform_3x3[:2, :].copy()
                    
                    # Adjust translation for region offsets
                    # Transform PCF region corners to see where they map in HE
                    pcf_tl_he = align_pcf_to_he_coordinates([[x_pcf_start, y_pcf_start]], transformation_matrix)[0]
                    
                    # The warp matrix should map HE output coordinates (relative to x_min_he, y_min_he)
                    # to PCF input coordinates (relative to x_pcf_start, y_pcf_start)
                    # Adjust translation accordingly
                    warp_matrix[0, 2] = warp_matrix[0, 2] - warp_matrix[0, 0] * x_min_he - warp_matrix[0, 1] * y_min_he + x_pcf_start
                    warp_matrix[1, 2] = warp_matrix[1, 2] - warp_matrix[1, 0] * x_min_he - warp_matrix[1, 1] * y_min_he + y_pcf_start
                    
                    # Apply warp
                    pcf_warped = cv2.warpAffine(
                        pcf_region,
                        warp_matrix,
                        (he_region.shape[1], he_region.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                    
                except ImportError:
                    # Fallback: use PIL with scale and rotation
                    from PIL import Image
                    
                    # Calculate scale and rotation
                    scale = np.sqrt(transformation_matrix[0,0]**2 + transformation_matrix[0,1]**2)
                    rotation_angle = np.arctan2(transformation_matrix[1,0], transformation_matrix[0,0]) * 180 / np.pi
                    
                    # Resize accounting for scale (inverse scale since we're going PCF -> HE)
                    target_width = int(pcf_region.shape[1] / scale) if scale > 0 else pcf_region.shape[1]
                    target_height = int(pcf_region.shape[0] / scale) if scale > 0 else pcf_region.shape[0]
                    
                    pcf_resized = Image.fromarray(pcf_region).resize(
                        (target_width, target_height), 
                        Image.Resampling.LANCZOS
                    )
                    
                    # Apply rotation (negative because we're inverting)
                    if abs(rotation_angle) > 0.1:
                        pcf_resized = pcf_resized.rotate(-rotation_angle, expand=False, resample=Image.Resampling.BILINEAR)
                    
                    # Resize to match HE region
                    pcf_warped = pcf_resized.resize(
                        (he_region.shape[1], he_region.shape[0]), 
                        Image.Resampling.LANCZOS
                    )
                    pcf_warped = np.array(pcf_warped)
                
                # Create overlay
                # Ensure both images have the same shape and data type
                if pcf_warped.shape[:2] != he_region.shape[:2]:
                    # Resize pcf_warped to match he_region
                    from PIL import Image
                    pcf_warped = Image.fromarray(pcf_warped).resize(
                        (he_region.shape[1], he_region.shape[0]), 
                        Image.Resampling.LANCZOS
                    )
                    pcf_warped = np.array(pcf_warped)
                
                # Ensure both images are uint8 in range [0, 255]
                if pcf_warped.dtype != np.uint8:
                    if pcf_warped.max() <= 1.0:
                        pcf_warped = (pcf_warped * 255).astype(np.uint8)
                    else:
                        pcf_warped = np.clip(pcf_warped, 0, 255).astype(np.uint8)
                
                if he_region.dtype != np.uint8:
                    if he_region.max() <= 1.0:
                        he_region = (he_region * 255).astype(np.uint8)
                    else:
                        he_region = np.clip(he_region, 0, 255).astype(np.uint8)
                
                # Check if pcf_warped has valid data
                pcf_mean = pcf_warped.mean()
                pcf_max = pcf_warped.max()
                he_mean = he_region.mean()
                
                # Create overlay with adjustable weights
                # Use the enhanced versions for overlay
                # Use 50-50 blend for better visibility
                overlay = (0.5 * he_display.astype(np.float32) + 0.5 * pcf_warped.astype(np.float32)).astype(np.uint8)
                
                # Ensure overlay is valid
                if overlay.size == 0 or overlay.shape[0] == 0 or overlay.shape[1] == 0:
                    ax_overlay.text(0.5, 0.5, f'Empty overlay\nHE: {he_region.shape}\nPCF: {pcf_warped.shape}', 
                           ha='center', va='center', transform=ax_overlay.transAxes, fontsize=8)
                    ax_overlay.axis('off')
                    continue
                
                # Debug info for first region
                if idx == 0:
                    print(f"  Region 1 debug:")
                    print(f"    PCF original: shape={pcf_region.shape}, mean={pcf_region.mean():.1f}, max={pcf_region.max()}, min={pcf_region.min()}")
                    print(f"    PCF warped: shape={pcf_warped.shape}, dtype={pcf_warped.dtype}, mean={pcf_mean:.1f}, max={pcf_max}, min={pcf_warped.min()}")
                    print(f"    HE region: shape={he_region.shape}, dtype={he_region.dtype}, mean={he_mean:.1f}, max={he_region.max()}, min={he_region.min()}")
                    print(f"    Overlay: shape={overlay.shape}, mean={overlay.mean():.1f}, max={overlay.max()}")
                
                # Display overlay
                ax_overlay.imshow(overlay)
                ax_overlay.set_title(f'R{idx+1} Overlay\n(50% HE + 50% PCF)', fontsize=10)
                ax_overlay.axis('off')
            except Exception as e:
                import traceback
                error_msg = str(e)
                print(f"Error in region {idx+1}: {error_msg}")
                if idx == 0:  # Only print full traceback for first error
                    print(traceback.format_exc())
                for ax in [ax_pcf_only, ax_he_only, ax_overlay]:
                    ax.text(0.5, 0.5, f'Error\n{error_msg[:40]}', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=7)
                    ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_regions * 3, len(axes)):
            axes[idx].axis('off')
        
        # plt.suptitle('Transformation Matrix Alignment Verification\n(Left: PCF, Middle: HE, Right: Overlay)', 
        plt.suptitle('Transformation Matrix Alignment Verification\n', 
                    fontsize=14, y=0.98)
        plt.tight_layout()
        
        # Ensure figure is displayed
        print(f"Created visualization with {n_regions} regions")
        print(f"Figure size: {fig.get_size_inches()}")
        
        he_slide.close()
        pcf_slide.close()

        if fig is not None:
            plt.show()
            print("✓ Visualization displayed")
            print("\nCheck the overlay images above:")
            print("  - If images are well-aligned: the overlay should show clear, sharp structures")
            print("  - If images are misaligned: you may see double/blurry structures")
        else:
            print("✗ Failed to create visualization")

        return fig  
        
    except Exception as e:
        print(f"Error verifying transformation: {e}")
        import traceback
        traceback.print_exc()
        if 'he_slide' in locals():
            try:
                he_slide.close()
            except:
                pass
        if 'pcf_slide' in locals():
            try:
                pcf_slide.close()
            except:
                pass
        return None




######################################################
# 2026.03.05 Get pixel size from server.json
######################################################
def get_pixel_size(segment_data_path, segment_number='1'):
    """
    Load pixel size from server.json file
    """
    server_json_path = os.path.join(segment_data_path, 'data', segment_number, 'server.json')
    if os.path.exists(server_json_path):
        try:
            with open(server_json_path, 'r') as f:
                server_data = json.load(f)
                pixel_width = server_data['metadata']['pixelCalibration']['pixelWidth']['value']
                pixel_height = server_data['metadata']['pixelCalibration']['pixelHeight']['value']
                # Usually width and height are the same, take average
                pixel_size_um = (pixel_width + pixel_height) / 2
                print(f"pixel_size_um read from server.json: {pixel_size_um:.6f} µm")

                width = server_data['metadata']['width']
                height = server_data['metadata']['height']
                print(f"width read from server.json: {width}")
                print(f"height read from server.json: {height}")

                return pixel_size_um
        except Exception as e:
            print(f"Failed to read server.json: {e}")
    return None


######################################################
# 2026.02.23 Coordinates transformation: PCF -> HE
######################################################
def add_pixel_coords(
    celltype_NCRT_tumor1,
    pixel_size_um_PCF,
    PCF2HE_transformation_NCRT,
    pixel_size_um_HE=None,
    sanity_check=True,
):
    # Check if columns exist
    if 'Centroid X µm' in celltype_NCRT_tumor1.columns and 'Centroid Y µm' in celltype_NCRT_tumor1.columns:
        print("="*60)
        print(f"\nConverting PCF_µm to PCF_pixel...")

        ## Add PCF pixel coordinate columns
        celltype_NCRT_tumor1['X_pix_PCF'] = celltype_NCRT_tumor1['Centroid X µm'] / pixel_size_um_PCF
        celltype_NCRT_tumor1['Y_pix_PCF'] = celltype_NCRT_tumor1['Centroid Y µm'] / pixel_size_um_PCF
        print(f"Added columns: X_pix_PCF, Y_pix_PCF")
        # print(f"X_pix_PCF range: [{celltype_NCRT_tumor1['X_pix_PCF'].min():.2f}, {celltype_NCRT_tumor1['X_pix_PCF'].max():.2f}]")
        # print(f"Y_pix_PCF range: [{celltype_NCRT_tumor1['Y_pix_PCF'].min():.2f}, {celltype_NCRT_tumor1['Y_pix_PCF'].max():.2f}]")

        ## Convert PCF_pixel to HE_pixel
        print(f"Converting PCF_pixel to HE_pixel...")
        pcf_coords = celltype_NCRT_tumor1[['X_pix_PCF', 'Y_pix_PCF']].values
        he_coords = align_pcf_to_he_coordinates(pcf_coords, PCF2HE_transformation_NCRT)
        celltype_NCRT_tumor1['X_pix_HE'] = he_coords[:, 0]
        celltype_NCRT_tumor1['Y_pix_HE'] = he_coords[:, 1]
        print(f"Added columns: X_pix_HE, Y_pix_HE")
        # print(f"X_pix_HE range: [{celltype_NCRT_tumor1['X_pix_HE'].min():.2f}, {celltype_NCRT_tumor1['X_pix_HE'].max():.2f}]")
        # print(f"Y_pix_HE range: [{celltype_NCRT_tumor1['Y_pix_HE'].min():.2f}, {celltype_NCRT_tumor1['Y_pix_HE'].max():.2f}]")

        if pixel_size_um_HE is not None:
            print("Converting HE_pixel to HE_µm...")
            celltype_NCRT_tumor1['X_um_HE'] = celltype_NCRT_tumor1['X_pix_HE'] * pixel_size_um_HE
            celltype_NCRT_tumor1['Y_um_HE'] = celltype_NCRT_tumor1['Y_pix_HE'] * pixel_size_um_HE
            print("Added columns: X_um_HE, Y_um_HE")

        ## Add a scale consistency check:
        ## Check the scale consistency of the PCF->HE transformation matrix:
        ## The scale of the linear part of the PCF->HE transformation matrix is compared with the pixel size ratio (pixel_size_PCF / pixel_size_HE),
        ## if the relative error is greater than 2%, a warning will be printed.
        if sanity_check and (pixel_size_um_HE is not None):
            A = np.array(PCF2HE_transformation_NCRT, dtype=float)[:, :2]
            scale = float(np.sqrt(A[0, 0] ** 2 + A[0, 1] ** 2))
            expected = float(pixel_size_um_PCF / pixel_size_um_HE)
            rel_err = abs(scale - expected) / expected if expected != 0 else np.nan
            if np.isfinite(rel_err) and rel_err > 0.02:
                print(
                    f"Warning: PCF->HE matrix scale ({scale:.6f}) deviates from "
                    f"pixel_size ratio pixel_size_PCF/pixel_size_HE ({expected:.6f}) by {rel_err*100:.2f}%."
                )

        ## Reorder columns
        existing_cols = [col for col in celltype_NCRT_tumor1.columns
                         if col not in ['X_pix_PCF', 'Y_pix_PCF', 'X_pix_HE', 'Y_pix_HE', 'X_um_HE', 'Y_um_HE']]
        # existing_cols = [col for col in celltype_NCRT_tumor1.columns
        #             if col not in ['X_pix_PCF', 'Y_pix_PCF', 'X_pix_HE', 'Y_pix_HE']]['Image', 'Object ID', 'Parent', 'Centroid X µm', 'Centroid Y µm', ]
        he_um_cols = ['X_um_HE', 'Y_um_HE'] if ('X_um_HE' in celltype_NCRT_tumor1.columns and 'Y_um_HE' in celltype_NCRT_tumor1.columns) else []
        new_column_order = existing_cols + ['X_pix_PCF', 'Y_pix_PCF', 'X_pix_HE', 'Y_pix_HE'] + he_um_cols
        celltype_NCRT_tumor1 = celltype_NCRT_tumor1[new_column_order]

        # print(f"\n4 pixel coordinate columns added at the end:")
        print(celltype_NCRT_tumor1.columns.tolist())
        return celltype_NCRT_tumor1
    else:
        print("Error: 'Centroid X µm' and 'Centroid Y µm' columns not found in celltype_NCRT_tumor1")
        print(f"Available columns: {celltype_NCRT_tumor1.columns.tolist()}")
        return celltype_NCRT_tumor1


######################################################
# 2026.02.26 Class distribution analysis and filtering
######################################################
def analyze_and_filter_classes_by_gap(X, y, ratio_threshold=1.5, abs_diff_threshold=100, figure_size=(12, 6),
                                       show_plots=True, verbose=True):
    """
    Analyze class distribution and filter classes based on automatic gap detection.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Class labels (cell types)
    ratio_threshold : float, default=1.5
        Minimum ratio for gap detection (current/previous)
    abs_diff_threshold : int, default=100
        Minimum absolute difference for gap detection
    show_plots : bool, default=True
        Whether to show visualization plots
    verbose : bool, default=True
        Whether to print detailed information
    
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'X_filtered': Filtered feature matrix
        - 'y_filtered': Filtered class labels
        - 'y_encoded': Encoded numeric labels
        - 'class_names': Class names after filtering
        - 'threshold_cell_type': Cell type used as threshold
        - 'threshold_count': Threshold count value
        - 'min_samples_per_class': Minimum samples per class (same as threshold_count)
        - 'class_counts': Class distribution before filtering
        - 'class_counts_filtered': Class distribution after filtering
    """
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    
    # Create copies to avoid modifying original data
    X = X.copy() if hasattr(X, 'copy') else np.array(X)
    y = y.copy() if hasattr(y, 'copy') else np.array(y)
    
    # Encode cell types to numeric labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    if verbose:
        print(f"  Encoded {len(class_names)} cell types")
    
    # Analyze class distribution
    class_counts = pd.Series(y).value_counts().sort_values(ascending=False)
    
    if verbose:
        print(f"\n  Class distribution (sorted by count):")
        print(class_counts)
    
    # Calculate statistics
    total_samples = len(y)
    num_classes = len(class_counts)
    max_count = class_counts.max()
    min_count = class_counts.min()
    median_count = class_counts.median()
    mean_count = class_counts.mean()
    
    if verbose:
        print(f"\n  Class distribution statistics:")
        print(f"    Total samples: {total_samples:,}")
        print(f"    Number of classes: {num_classes}")
        print(f"    Max samples per class: {max_count:,} ({class_counts.idxmax()})")
        print(f"    Min samples per class: {min_count:,} ({class_counts.idxmin()})")
        print(f"    Median samples per class: {median_count:.1f}")
        print(f"    Mean samples per class: {mean_count:.1f}")
        print(f"    Imbalance ratio (max/min): {max_count/min_count:.1f}:1")
    
    # Determine filtering strategy: Find the first large gap from right to left (small to large)
    class_counts_sorted_asc = class_counts.sort_values(ascending=True)
    
    if verbose:
        print(f"\n  Analyzing gaps from smallest to largest (right to left):")
        print(f"    Total classes: {len(class_counts_sorted_asc)}")
    
    # Calculate relative changes (ratio) and absolute differences between adjacent values
    gap_info = []
    for i in range(1, len(class_counts_sorted_asc)):
        prev_name = class_counts_sorted_asc.index[i-1]
        curr_name = class_counts_sorted_asc.index[i]
        prev_count = class_counts_sorted_asc.iloc[i-1]
        curr_count = class_counts_sorted_asc.iloc[i]
        if prev_count > 0:
            ratio = curr_count / prev_count
            abs_diff = curr_count - prev_count
            gap_info.append((prev_name, curr_name, prev_count, curr_count, ratio, abs_diff))
    
    # Print top gaps for debugging
    if verbose and len(gap_info) > 0:
        gap_info_sorted = sorted(gap_info, key=lambda x: x[4], reverse=True)
        print(f"\n  Top 5 largest gaps (by ratio):")
        for idx, (prev_name, curr_name, prev_count, curr_count, ratio, abs_diff) in enumerate(gap_info_sorted[:5]):
            print(f"    {idx+1}. {prev_name} ({prev_count}) -> {curr_name} ({curr_count}), ratio: {ratio:.2f}, diff: {abs_diff}")
    
    # Find the first large gap from right to left
    gap_found = False
    threshold_cell_type = None
    threshold_count = None
    
    for prev_name, curr_name, prev_count, curr_count, ratio, abs_diff in gap_info:
        if ratio >= ratio_threshold and abs_diff >= abs_diff_threshold:
            threshold_cell_type = prev_name
            threshold_count = int(prev_count)
            gap_found = True
            if verbose:
                print(f"\n  ✓ Found large gap: {prev_name} ({prev_count}) -> {curr_name} ({curr_count})")
                print(f"    Ratio: {ratio:.2f}, Absolute difference: {abs_diff}")
            break
    
    if not gap_found:
        if verbose:
            print(f"\n  No gap found with ratio >= {ratio_threshold} and diff >= {abs_diff_threshold}")
            print(f"  Trying with lower thresholds...")
        
        # Try with just ratio threshold
        for prev_name, curr_name, prev_count, curr_count, ratio, abs_diff in gap_info:
            if ratio >= 1.3:
                threshold_cell_type = prev_name
                threshold_count = int(prev_count)
                gap_found = True
                if verbose:
                    print(f"  ✓ Found gap with lower threshold: {prev_name} ({prev_count}) -> {curr_name} ({curr_count})")
                    print(f"    Ratio: {ratio:.2f}, Absolute difference: {abs_diff}")
                break
        
        if not gap_found:
            # Fallback: use a percentile-based threshold
            percentile_10_value = int(np.percentile(class_counts_sorted_asc.values, 10))
            closest_idx = (class_counts_sorted_asc.values - percentile_10_value).argmin()
            threshold_cell_type = class_counts_sorted_asc.index[closest_idx]
            threshold_count = int(class_counts_sorted_asc.iloc[closest_idx])
            if verbose:
                print(f"\n  Using 10th percentile-based threshold: {threshold_cell_type} ({threshold_count})")
    
    min_samples_per_class = threshold_count
    
    if verbose:
        print(f"\n  Filtering Strategy (Automatic Gap Detection):")
        print(f"    Threshold cell type: {threshold_cell_type}")
        print(f"    Threshold count: {min_samples_per_class}")
        print(f"    All cell types with count < {min_samples_per_class} will be filtered out")
    
    # Calculate 10th percentile
    percentile_10 = np.percentile(class_counts.values, 10)
    
    # Visualize class distribution
    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=figure_size)
        
        # Plot 1: Bar chart of all classes with cell type names on x-axis
        x_pos = np.arange(len(class_counts))
        axes[0].bar(x_pos, class_counts.values, color='steelblue', alpha=0.7)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(class_counts.index, rotation=90, ha='center', fontsize=8)
        axes[0].set_xlabel('Cell Type', fontsize=12)
        axes[0].set_ylabel('Number of Samples', fontsize=12)
        axes[0].set_title('Class Distribution (All Classes)', fontsize=14)
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        axes[0].axhline(y=mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.0f}')
        axes[0].axhline(y=median_count, color='green', linestyle='--', linewidth=2, label=f'Median: {median_count:.0f}')
        axes[0].axhline(y=percentile_10, color='orange', linestyle='--', linewidth=2, label=f'10th Percentile: {percentile_10:.0f}')
        axes[0].axhline(y=min_samples_per_class, color='red', linestyle='-', linewidth=2, label=f'Gap Threshold: {threshold_cell_type} ({min_samples_per_class})')
        axes[0].legend()
        
        # Plot 2: Only show filtered cell types
        valid_classes_for_plot = class_counts[class_counts >= min_samples_per_class]
        x_pos_filtered = np.arange(len(valid_classes_for_plot))
        
        axes[1].bar(x_pos_filtered, valid_classes_for_plot.values, color='green', alpha=0.7)
        axes[1].set_xticks(x_pos_filtered)
        axes[1].set_xticklabels(valid_classes_for_plot.index, rotation=90, ha='center', fontsize=8)
        axes[1].set_xlabel('Cell Type', fontsize=12)
        axes[1].set_ylabel('Number of Samples', fontsize=12)
        axes[1].set_title(f'Filtered Class Distribution ({len(valid_classes_for_plot)} classes, ≥{min_samples_per_class} samples)', fontsize=14)
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1].axhline(y=min_samples_per_class, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Threshold: {min_samples_per_class}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    # Apply filtering
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    filtered_classes = class_counts[class_counts < min_samples_per_class].index
    
    if verbose:
        print(f"\n  Filtering results:")
        print(f"    Classes to keep: {len(valid_classes)}")
        print(f"    Classes to filter out: {len(filtered_classes)}")
        if len(filtered_classes) > 0:
            print(f"    Filtered classes (< {min_samples_per_class} samples):")
            for cls in filtered_classes:
                print(f"      - {cls}: {class_counts[cls]} samples")
    
    valid_mask = pd.Series(y).isin(valid_classes)
    
    if valid_mask.sum() < len(y):
        if verbose:
            print(f"\n  Applying filtering (removing classes with < {min_samples_per_class} samples)...")
            print(f"  Before filtering: {len(y):,} samples, {len(class_counts)} classes")
        
        X = X[valid_mask.values]
        y = y[valid_mask.values]
        
        # Re-encode labels after filtering
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_
        
        # Recalculate statistics after filtering
        class_counts_filtered = pd.Series(y).value_counts().sort_values(ascending=False)
        max_count_filtered = class_counts_filtered.max()
        min_count_filtered = class_counts_filtered.min()
        median_count_filtered = class_counts_filtered.median()
        mean_count_filtered = class_counts_filtered.mean()
        
        if verbose:
            print(f"  After filtering: {len(y):,} samples, {len(class_names)} classes")
            print(f"  Remaining class statistics:")
            print(f"    Max: {max_count_filtered:,} ({class_counts_filtered.idxmax()})")
            print(f"    Min: {min_count_filtered:,} ({class_counts_filtered.idxmin()})")
            print(f"    Median: {median_count_filtered:.1f}")
            print(f"    Mean: {mean_count_filtered:.1f}")
            print(f"    New imbalance ratio: {max_count_filtered/min_count_filtered:.1f}:1")
        
        # Visualize filtered distribution
        # if show_plots:
        #     fig, axes = plt.subplots(1, 2, figsize=figure_size)
            
        #     axes[0].bar(range(len(class_counts_filtered)), class_counts_filtered.values, color='green', alpha=0.7)
        #     axes[0].set_xlabel('Class Index (sorted by count)', fontsize=12)
        #     axes[0].set_ylabel('Number of Samples', fontsize=12)
        #     axes[0].set_title(f'Class Distribution After Filtering ({len(class_names)} classes)', fontsize=14)
        #     axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        #     axes[0].axhline(y=mean_count_filtered, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count_filtered:.0f}')
        #     axes[0].axhline(y=median_count_filtered, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_count_filtered:.0f}')
        #     axes[0].legend()
            
        #     # Show class names for top classes
        #     top_n = min(10, len(class_counts_filtered))
        #     axes[1].barh(range(top_n), class_counts_filtered.head(top_n).values, color='steelblue', alpha=0.7)
        #     axes[1].set_yticks(range(top_n))
        #     axes[1].set_yticklabels(class_counts_filtered.head(top_n).index, fontsize=9)
        #     axes[1].set_xlabel('Number of Samples', fontsize=12)
        #     axes[1].set_title(f'Top {top_n} Classes (After Filtering)', fontsize=14)
        #     axes[1].grid(axis='x', alpha=0.3, linestyle='--')
            
        #     plt.tight_layout()
        #     plt.show()
    else:
        if verbose:
            print(f"\n  No filtering needed - all classes have >= {min_samples_per_class} samples")
        class_counts_filtered = class_counts
    
    # Return results
    return {
        'X_filtered': X,
        'y_filtered': y,
        'y_encoded': y_encoded,
        'class_names': class_names,
        'threshold_cell_type': threshold_cell_type,
        'threshold_count': threshold_count,
        'min_samples_per_class': min_samples_per_class,
        'class_counts': class_counts,
        'class_counts_filtered': class_counts_filtered
    }



#########################################
# 2026.02.26 LLY Create PyTorch Datasets and DataLoaders
#########################################
from torch.utils.data import Dataset, DataLoader
import torch

class CellTypeDataset(Dataset):
    """
    Custom PyTorch Dataset for cell type classification.
    """
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        # Ensure labels are integer type before converting to LongTensor
        if isinstance(labels, np.ndarray):
            if labels.dtype.kind in ['U', 'S', 'O']:  # String type
                raise ValueError(f"Labels must be numeric, but got dtype: {labels.dtype}. Please encode labels to integers before creating dataset.")
            labels = labels.astype(np.int64)
        self.labels = torch.LongTensor(labels)
        # Ensure features and labels have the same length
        if len(self.features) != len(self.labels):
            min_len = min(len(self.features), len(self.labels))
            print(f"  ⚠ Warning: features ({len(self.features)}) and labels ({len(self.labels)}) length mismatch. Truncating to {min_len}")
            self.features = self.features[:min_len]
            self.labels = self.labels[:min_len]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Add bounds checking to prevent IndexError
        if idx >= len(self.features):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.features)}")
        if idx < 0:
            idx = len(self.features) + idx  # Handle negative indexing
            if idx < 0:
                raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.features)}")
        return self.features[idx], self.labels[idx]

def create_dataloaders(
    X_train, y_train, X_val, y_val,
    batch_size_gpu=1024, batch_size_cpu=256
):
    """
    Create PyTorch Datasets and DataLoaders for training and validation.

    Args:
        X_train (np.ndarray or tensor): Training feature data
        y_train (np.ndarray or tensor): Training labels
        X_val (np.ndarray or tensor): Validation feature data
        y_val (np.ndarray or tensor): Validation labels
        batch_size_gpu (int): Batch size if CUDA is available
        batch_size_cpu (int): Batch size for CPU

    Returns:
        train_dataset, val_dataset, train_loader, val_loader
    """
    # Create datasets
    train_dataset = CellTypeDataset(X_train, y_train)
    val_dataset = CellTypeDataset(X_val, y_val)

    # Determine batch size and DataLoader settings based on device
    use_cuda = torch.cuda.is_available()
    batch_size = batch_size_gpu if use_cuda else batch_size_cpu
    num_workers = 4 if use_cuda else 0
    pin_memory = True if use_cuda else False

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Feature dimension: {X_train.shape[1]}")

    return train_dataset, val_dataset, train_loader, val_loader


######################################
# 2026.02.26 LLY MLP model training 
######################################
import torch
import torch.nn as nn
import torch.nn.functional as F_nn

########################################################
# 2026.06.26 For each dataset, add spatial context for model training and validation
#            每个 cell 进入 MLP 的向量 = 自身 UNI embedding + 邻居上下文的线性投影
#            concat 完全可行，只是当前实现选择了 residual add, 主要出于这些考虑：
#            (a) 训练稳定性：零初始化 residual; 
# (若用 concat [x_i; ctx] → Linear(2048, 512)，第一层参数随机初始化，一开始就和非 spatial 模型完全不同，更难稳定训练，也 harder 做 fair comparison)
########################################################
class SpatialContextFusion(nn.Module):
    """
    Optional neighbor-context fusion on cell embeddings (before MLP backbone).

    Residual form ``x + proj(aggregate(neighbors))`` with zero-init ``proj`` so
    training starts near the non-spatial model. When ``enabled=False`` or
    ``neighbor_x is None``, returns ``x`` unchanged.
    """

    def __init__(self, dim, mode="mean", dropout=0.1, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)
        self.mode = str(mode)
        if self.mode == "attention":
            self.attn_q = nn.Linear(dim, dim)
            self.attn_k = nn.Linear(dim, dim)
            self.scale = dim ** -0.5
        elif self.mode != "mean":
            raise ValueError("spatial_mode must be one of: mean, attention")
        self.proj = nn.Linear(dim, dim)    # 可学习的 Linear(1024 → 1024)，用于将邻居的 UNI embedding 融合到当前样本的 embedding
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    ########################################################
    # 2026.06.26 x_fused = x_i + proj(ctx)
    ########################################################
    def forward(self, x, neighbor_x=None):
        if not self.enabled or neighbor_x is None:
            return x
        if self.mode == "mean":
            ctx = neighbor_x.mean(dim=1)    # The UNI embeddings of each of the k neighbors, with shape (K, 1024)
        else:
            q = self.attn_q(x).unsqueeze(1)
            k = self.attn_k(neighbor_x)
            w = torch.softmax((q * k).sum(dim=-1) * self.scale, dim=1)
            ctx = (w.unsqueeze(-1) * neighbor_x).sum(dim=1)
        return x + self.dropout(self.proj(ctx))    # still 1024 dim, used before MLP


def build_spatial_neighbor_index(coords, k_neighbors=8):
    """
    kNN neighbor indices per cell (excluding self).

    Returns ``(N, k)`` int64 array with ``k = min(k_neighbors, N-1)``.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must be (N, 2+), got {coords.shape}")
    n = coords.shape[0]
    if n <= 1:
        return np.zeros((n, 1), dtype=np.int64)
    k = min(int(k_neighbors), n - 1)
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn_model.fit(coords)
    _, idx = nn_model.kneighbors(coords)
    return idx[:, 1:].astype(np.int64)


########################################################
# 2026.06.26 For each dataset, add spatial context for model training and validation
########################################################
def build_spatial_neighbor_index_by_group(coords, groups, k_neighbors=8):
    """
    kNN neighbor indices per cell, computed **within each group** (e.g. sample).

    Pooled cross-dataset data must not use a single global kNN on ``spatial_HE``,
    because each sample has its own coordinate system. This builds kNN per group,
    maps local row ids back to global indices, and returns ``(N, k_neighbors)``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    groups = np.asarray(groups)
    if coords.shape[0] != groups.shape[0]:
        raise ValueError(
            f"coords and groups length mismatch: {coords.shape[0]} vs {groups.shape[0]}"
        )
    n = coords.shape[0]
    if n == 0:
        return np.zeros((0, int(k_neighbors)), dtype=np.int64)

    k_cols = int(k_neighbors)
    out = np.zeros((n, k_cols), dtype=np.int64)
    for g in np.unique(groups):
        mask = groups == g
        idx_global = np.flatnonzero(mask)
        sub_nbr_local = build_spatial_neighbor_index(coords[mask], k_neighbors=k_neighbors)
        k_sub = sub_nbr_local.shape[1]
        out[mask, :k_sub] = idx_global[sub_nbr_local]
        if k_sub < k_cols:
            out[mask, k_sub:] = idx_global[sub_nbr_local[:, :1]]
    return out
########################################################


def gather_neighbor_embeddings(X, neighbor_index, row_indices, device=None):
    """
    Gather neighbor embedding blocks for batch rows.

    Parameters
    ----------
    X : (N, D) tensor — full scaled embedding matrix (same row order as neighbor_index).
    neighbor_index : (N, K) long tensor.
    row_indices : (B,) long tensor — global row ids for the batch.
    device : torch.device or str, optional
        Align all tensors to this device before ``index_select``. When omitted,
        uses ``row_indices.device``.
    """
    if row_indices is None:
        return None
    if device is not None:
        dev = device if isinstance(device, torch.device) else torch.device(device)
    else:
        dev = row_indices.device
    if row_indices.device != dev:
        row_indices = row_indices.to(dev, non_blocking=True)
    if neighbor_index.device != dev:
        neighbor_index = neighbor_index.to(dev, non_blocking=True)
    if X.device != dev:
        X = X.to(dev, non_blocking=True)
    nbr_idx = neighbor_index.index_select(0, row_indices)
    return X.index_select(0, nbr_idx.reshape(-1)).reshape(
        row_indices.shape[0], nbr_idx.shape[1], X.shape[1]
    )


# Define Improved MLP Model with Residual Connections
class ImprovedMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim=384,
        num_classes=42,
        hidden_dims=[1024, 512, 256],
        dropout=0.2,
        use_spatial_context=False,
        spatial_mode="mean",
        spatial_k=8,
    ):
        super().__init__()
        self.use_spatial_context = bool(use_spatial_context)
        self.spatial_k = int(spatial_k)
        self.spatial_fusion = (
            SpatialContextFusion(input_dim, mode=spatial_mode, dropout=dropout, enabled=True)
            if self.use_spatial_context
            else None
        )

        # First layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),  # LayerNorm instead of BatchNorm for better stability
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _apply_spatial(self, x, neighbor_x=None):
        if self.spatial_fusion is not None and neighbor_x is not None:
            return self.spatial_fusion(x, neighbor_x)
        return x

    def forward(self, x, neighbor_x=None):
        x = self._apply_spatial(x, neighbor_x)
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x


class DualHeadImprovedMLPClassifier(ImprovedMLPClassifier):
    """
    MLP with two supervised heads:
    - level2 head (fine classes)
    - level1 head (coarse classes)

    ``forward`` keeps backward compatibility by returning only level2 logits.
    Use ``forward_heads`` when training with dual-head objectives.
    """

    def __init__(
        self,
        input_dim=384,
        num_classes=42,
        num_level1_classes=7,
        hidden_dims=[1024, 512, 256],
        dropout=0.2,
        use_spatial_context=False,
        spatial_mode="mean",
        spatial_k=8,
    ):
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_spatial_context=use_spatial_context,
            spatial_mode=spatial_mode,
            spatial_k=spatial_k,
        )
        self.level1_head = nn.Linear(hidden_dims[-1], num_level1_classes)
        nn.init.kaiming_normal_(self.level1_head.weight, mode="fan_out", nonlinearity="relu")
        if self.level1_head.bias is not None:
            nn.init.constant_(self.level1_head.bias, 0)

    def _forward_backbone(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return x

    def forward_heads(self, x, neighbor_x=None):
        x = self._apply_spatial(x, neighbor_x)
        feat = self._forward_backbone(x)
        logits_l2 = self.output_layer(feat)
        logits_l1 = self.level1_head(feat)
        return logits_l2, logits_l1

    def forward(self, x, neighbor_x=None):
        logits_l2, _ = self.forward_heads(x, neighbor_x=neighbor_x)
        return logits_l2


class FiveHeadImprovedMLPClassifier(DualHeadImprovedMLPClassifier):
    """
    Dual-head MLP plus three auxiliary heads for Xenium multi-label training:

    - ``level12_head``: celltype_level12 (final_sublineage)
    - ``level3_head``: celltype_level3 (CNiche)
    - ``level4_head``: celltype_level4 (TNiche)

    ``forward`` still returns level2 logits only (StarDist / legacy evaluate).
    Use ``forward_heads`` for all five supervised heads.
    """

    def __init__(
        self,
        input_dim=384,
        num_classes=42,
        num_level1_classes=7,
        num_level12_classes=20,
        num_level3_classes=4,
        num_level4_classes=8,
        hidden_dims=(1024, 512, 256),
        dropout=0.2,
        use_spatial_context=False,
        spatial_mode="mean",
        spatial_k=8,
    ):
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            num_level1_classes=num_level1_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_spatial_context=use_spatial_context,
            spatial_mode=spatial_mode,
            spatial_k=spatial_k,
        )
        last_dim = hidden_dims[-1]
        self.level12_head = nn.Linear(last_dim, num_level12_classes)
        self.level3_head = nn.Linear(last_dim, num_level3_classes)
        self.level4_head = nn.Linear(last_dim, num_level4_classes)
        for head in (self.level12_head, self.level3_head, self.level4_head):
            nn.init.kaiming_normal_(head.weight, mode="fan_out", nonlinearity="relu")
            if head.bias is not None:
                nn.init.constant_(head.bias, 0)

    def forward_heads(self, x, neighbor_x=None):
        x = self._apply_spatial(x, neighbor_x)
        feat = self._forward_backbone(x)
        logits_l2 = self.output_layer(feat)
        logits_l1 = self.level1_head(feat)
        logits_l12 = self.level12_head(feat)
        logits_l3 = self.level3_head(feat)
        logits_l4 = self.level4_head(feat)
        return logits_l2, logits_l1, logits_l12, logits_l3, logits_l4

    def forward(self, x, neighbor_x=None):
        """Legacy ``evaluate()`` / StarDist path: level2 logits only."""
        logits_l2, _, _, _, _ = self.forward_heads(x, neighbor_x=neighbor_x)
        return logits_l2


def forward_heads_l2_l1(model, x, neighbor_x=None):
    """Return ``(logits_l2, logits_l1)`` for dual- or five-head models."""
    out = model.forward_heads(x, neighbor_x=neighbor_x)
    return out[0], out[1]


#########################################
# 2026.02.26 LLY Train MLP model
#########################################

import time
import torch

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    evaluate,
    device,
    scaler=None,
    patience=10,
    max_epochs=50,
    save_path=None
):
    """
    Train a PyTorch model with early stopping, gradient clipping, and optional AMP.

    Args:
        model: PyTorch model to be trained.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        evaluate: Function to evaluate model, should return (acc, macro_f1, weighted_f1, preds, labels).
        device: torch.device('cuda') or torch.device('cpu').
        scaler: torch.amp.GradScaler("cuda") instance or None (AMP on CUDA).
        path: Path to save the best model.
        therapy_data: Additional info for filename construction.
        patience: Early stopping patience.
        max_epochs: Maximum number of epochs.
        save_filename: Name of the best model file to save.
    Returns:
        best_weighted_f1, best_macro_f1, best_epoch
    """
    best_f1 = 0
    best_weighted_f1 = 0
    counter = 0
    best_epoch = 0

    print("Starting training with improved model...")
    print("="*60)

    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            num_batches += 1

        # Validation
        val_acc, val_macro_f1, val_weighted_f1, val_preds, val_labels = evaluate(model, val_loader, device, scaler)
        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1}/{max_epochs} ({epoch_time:.2f}s, LR: {current_lr:.6f})")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Val Macro-F1: {val_macro_f1:.4f}")
        print(f"  Val Weighted-F1: {val_weighted_f1:.4f}")

        # Early stopping based on weighted F1
        if val_weighted_f1 > best_weighted_f1:
            best_weighted_f1 = val_weighted_f1
            best_f1 = val_macro_f1
            best_epoch = epoch + 1
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
            counter = 0
            print(f"  ✓ Best model saved (Weighted-F1: {best_weighted_f1:.4f}, Macro-F1: {best_f1:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best Weighted-F1: {best_weighted_f1:.4f} at epoch {best_epoch}")
                print(f"Best Macro-F1: {best_f1:.4f} at epoch {best_epoch}")
                break

    print("="*60)
    print("Training completed!")
    print(f"Best validation Weighted-F1: {best_weighted_f1:.4f} at epoch {best_epoch}")
    print(f"Best validation Macro-F1: {best_f1:.4f} at epoch {best_epoch}")

    return best_weighted_f1, best_f1, best_epoch



######################################
# 2026.02.26 LLY Load Vit embeddings
######################################
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch


def _pth_embedding_payload_to_numpy(embedding_data):
    """Convert torch.load() payload from a HIPT/ViT-style .pth file to a 1D numpy vector."""
    if isinstance(embedding_data, tuple):
        tensor_list = []
        for t in embedding_data:
            if isinstance(t, torch.Tensor):
                t_np = t.detach().cpu().numpy()
                tensor_list.append(t_np.flatten())
            else:
                tensor_list.append(np.array(t).flatten())
        embedding = np.concatenate(tensor_list)
    elif isinstance(embedding_data, torch.Tensor):
        embedding = embedding_data.detach().cpu().numpy()
        if embedding.ndim == 0:
            embedding = np.array([embedding.item()])
        if embedding.ndim > 1:
            embedding = embedding.flatten()
    elif isinstance(embedding_data, list):
        tensor_list = []
        for item in embedding_data:
            if isinstance(item, torch.Tensor):
                tensor_list.append(item.detach().cpu().numpy().flatten())
            else:
                tensor_list.append(np.array(item).flatten())
        embedding = np.concatenate(tensor_list)
    else:
        try:
            embedding = np.array(embedding_data)
            if embedding.ndim == 0:
                embedding = np.array([embedding.item()])
            if embedding.ndim > 1:
                embedding = embedding.flatten()
        except Exception as e:
            raise ValueError(f"Cannot convert embedding to numpy array: {e}") from e

    if embedding.ndim > 1:
        embedding = embedding.flatten()
    return embedding


def _load_pth_embedding_file(pth_path):
    embedding_data = torch.load(pth_path, map_location="cpu")
    return _pth_embedding_payload_to_numpy(embedding_data)


def _pth_coord_regex(pth_prefix: str):
    """Full-string match (only for basenames that are exactly ``prefix_x_y.pth``)."""
    return re.compile(rf"^{re.escape(pth_prefix)}_([\d.]+)_([\d.]+)\.pth$")


_PTH_FLAT_BASENAME = re.compile(r"^(.+)_([\d.]+)_([\d.]+)\.pth$")


def infer_pth_prefix_from_dir(embedding_dir, max_sample: int = 64):
    """
    Infer ``prefix`` from flat ``{prefix}_{x}_{y}.pth`` basenames in *embedding_dir*.

    Complete_Cases Xenium exports often use ``VUILD107MA_x_y.pth`` while Select4 /
    CODEX use ``sc_{sample}_x_y.pth``. Call when ``pth_prefix`` loads zero files.
    """
    embedding_dir = Path(embedding_dir)
    if not embedding_dir.is_dir():
        return None
    counts: dict[str, int] = {}
    seen = 0
    with os.scandir(embedding_dir) as it:
        for entry in it:
            if not entry.is_file() or not entry.name.endswith(".pth"):
                continue
            m = _PTH_FLAT_BASENAME.match(entry.name)
            if not m:
                continue
            prefix = m.group(1)
            counts[prefix] = counts.get(prefix, 0) + 1
            seen += 1
            if seen >= max_sample:
                break
    if not counts:
        return None
    return max(counts, key=counts.get)


def _pth_coord_search_regex(pth_prefix: str):
    """
    Substring search (legacy ``load_embeddings`` behavior).

    Matches ``prefix_x_y.pth`` anywhere in the filename, e.g. long crop-based names.
    """
    return re.compile(rf"{re.escape(pth_prefix)}_([\d.]+)_([\d.]+)\.pth")


def _parse_coords_from_pth_filename(filename, search_pat):
    m = search_pat.search(filename)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _selective_embedding_probe_mode(
    embedding_dir: Path,
    pth_prefix: str,
    search_pat,
    max_sample: int = 512,
) -> tuple:
    """
    One scandir pass to decide how selective basename resolution should run.

    Returns
    -------
    (mode, detail)
        mode ``abort`` — flat ``prefix_x_y.pth`` probes cannot work (e.g. only long
        crop stems); return empty and let caller full-scan.
        mode ``raw_unique`` — basenames look like ``sc_NCRT_10000.01…_6157.05….pth``;
        use only unique cell (x,y) as probes (fast).
        mode ``legacy`` — mixed / integer stems; keep lattice + rounded-decimal probes.
    """
    flat_pat = _pth_coord_regex(pth_prefix)
    embedding_dir = Path(embedding_dir)
    n_sub = 0
    n_flat = 0
    n_flat_frac = 0
    scanned = 0
    with os.scandir(embedding_dir) as it:
        for e in it:
            if not e.is_file() or not e.name.endswith(".pth"):
                continue
            scanned += 1
            name = e.name
            m = flat_pat.match(name)
            if m:
                n_flat += 1
                g1, g2 = m.group(1), m.group(2)
                if "." in g1 and "." in g2:
                    n_flat_frac += 1
            elif search_pat.search(name) is not None:
                n_sub += 1
            if scanned >= max_sample:
                break

    if scanned == 0:
        return "abort", "no_pth_in_dir"
    if n_flat == 0 and n_sub == 0:
        return "abort", "no_regex_match_in_sample"
    if n_flat == 0 and n_sub > 0:
        return "abort", "substring_only_long_names"
    if n_flat > 0 and (n_flat_frac / max(n_flat, 1)) >= 0.85:
        return "raw_unique", f"flat_fractional_stems ({n_flat_frac}/{n_flat} in sample)"
    return "legacy", f"mixed_or_integer_stems ({n_flat_frac}/{n_flat} fractional in sample)"


def _resolve_cell_coord_columns(celltype_df, coord_cols=None):
    """Same column resolution rules as match_embeddings."""
    if coord_cols is not None:
        x_col, y_col = coord_cols
        if x_col not in celltype_df.columns or y_col not in celltype_df.columns:
            raise ValueError(
                f"Specified columns {x_col} or {y_col} not found in celltype_df"
            )
        return x_col, y_col
    if "centroid_x" in celltype_df.columns and "centroid_y" in celltype_df.columns:
        return "centroid_x", "centroid_y"
    if "X_pix_HE" in celltype_df.columns and "Y_pix_HE" in celltype_df.columns:
        return "X_pix_HE", "Y_pix_HE"
    raise ValueError(
        "celltype_df must contain either ['centroid_x', 'centroid_y'] or "
        "['X_pix_HE', 'Y_pix_HE'] columns, or pass coord_cols=..."
    )


def _grid_coords_within_tolerance(cx, cy, tolerance):
    """Per-cell candidate patch keys (legacy slow path)."""
    tol = float(tolerance)
    cx, cy = float(cx), float(cy)
    seen = set()
    out = []

    def add(ex, ey):
        k = (round(float(ex), 6), round(float(ey), 6))
        if k not in seen:
            seen.add(k)
            out.append((float(ex), float(ey)))

    if tol <= 0:
        add(cx, cy)
        return out

    x0 = int(np.floor(cx - tol - 1e-9))
    x1 = int(np.ceil(cx + tol + 1e-9))
    y0 = int(np.floor(cy - tol - 1e-9))
    y1 = int(np.ceil(cy + tol + 1e-9))
    tol2 = tol * tol
    for xi in range(x0, x1 + 1):
        for yi in range(y0, y1 + 1):
            dx = xi - cx
            dy = yi - cy
            if dx * dx + dy * dy <= tol2 + 1e-12:
                add(xi, yi)

    add(cx, cy)
    add(round(cx, 2), round(cy, 2))
    add(round(cx, 4), round(cy, 4))
    return out


def _pth_coordinate_stem_variants(v: float):
    """
    Text fragments used in ``{prefix}_{x}_{y}.pth`` stems.

    On-disk names often use NumPy-style positional formatting; ``str(float)`` can
    differ (e.g. from ``10000.018341876836``), so we try a few equivalent forms.
    """
    v = float(np.float64(v))
    out = []
    seen = set()

    def push(s):
        if s not in seen:
            seen.add(s)
            out.append(s)

    push(str(v))
    push(np.format_float_positional(v, unique=True, trim="k"))
    push(np.format_float_positional(v, precision=17, unique=False, trim="k"))
    return out


def _candidate_pth_basenames(pth_prefix: str, ex: float, ey: float):
    basenames = []
    seen = set()

    def add(name):
        if name not in seen:
            seen.add(name)
            basenames.append(name)

    ix, iy = int(round(ex)), int(round(ey))
    if abs(ex - ix) < 1e-6 and abs(ey - iy) < 1e-6:
        add(f"{pth_prefix}_{ix}_{iy}.pth")

    for sx in _pth_coordinate_stem_variants(ex):
        for sy in _pth_coordinate_stem_variants(ey):
            add(f"{pth_prefix}_{sx}_{sy}.pth")

    if float(ex).is_integer() and float(ey).is_integer():
        add(f"{pth_prefix}_{int(ex)}_{int(ey)}.pth")
    return basenames


def _unique_patch_lattice_indices(xv, yv, tolerance):
    """Vectorized unique patch indices + dilation within tolerance (fast path)."""
    tol = float(tolerance)
    ic = np.stack(
        [np.rint(xv.astype(np.float64)), np.rint(yv.astype(np.float64))], axis=1
    ).astype(np.int64)
    ic = np.unique(ic, axis=0)
    if ic.size == 0:
        return ic

    R = max(0, int(np.ceil(tol)))
    if R == 0:
        return ic

    grid_d = np.arange(-R, R + 1)
    DX, DY = np.meshgrid(grid_d, grid_d, indexing="ij")
    dist2 = DX.astype(np.float64) ** 2 + DY.astype(np.float64) ** 2
    msk = dist2 <= (tol + 1e-9) ** 2
    offs = np.column_stack([DX[msk], DY[msk]]).astype(np.int64)
    expanded = ic[:, None, :] + offs[None, :, :]
    return np.unique(expanded.reshape(-1, 2), axis=0)


def _merge_batch_probe_coords(xv, yv, tolerance, log_parts=False):
    """
    Combine dilated integer lattice with rounded *actual* cell coordinates.

    Filenames often use decimals (e.g. ``..._1234.56_7890.12.pth``); lattice-only
    probes can miss them entirely.

    The returned row count is typically **much larger** than ``len(xv)`` because:
    (1) each distinct rounded-pixel anchor is dilated by ~O(tolerance²) integer
    offsets to match integer-stem .pth names; (2) unique 1/2/4-decimal-roundings
    of *every* cell are unioned. This set is only for filesystem path resolution,
    not "one probe per cell".
    """
    lat = _unique_patch_lattice_indices(xv, yv, tolerance).astype(np.float64)
    blocks = [lat]
    if log_parts:
        print(f"    [probe breakdown] dilated integer lattice: {len(lat):,}")
    for nd in (1, 2, 4):
        dec = np.unique(
            np.column_stack([np.round(xv.astype(np.float64), nd), np.round(yv.astype(np.float64), nd)]),
            axis=0,
        )
        blocks.append(dec.astype(np.float64))
        if log_parts:
            print(f"    [probe breakdown] unique coords rounded to {nd} dp: {len(dec):,}")
    merged = np.vstack(blocks)
    out = np.unique(merged, axis=0)
    if log_parts:
        print(f"    [probe breakdown] union (unique rows): {len(out):,}")
    return out


def _merge_batch_probe_coords_raw_unique(xv, yv, log_parts=False):
    """Unique float64 (x,y) from cells — enough when .pth basenames use those coordinates."""
    raw = np.unique(
        np.column_stack([xv.astype(np.float64), yv.astype(np.float64)]),
        axis=0,
    ).astype(np.float64)
    if log_parts:
        print(f"    [probe breakdown] unique raw cell (x,y): {len(raw):,}")
    return raw


def _collect_selective_pth_paths(embedding_dir, probe_xy, pth_prefix, search_pat):
    seen = set()
    paths = []
    embedding_dir = Path(embedding_dir)
    for row in tqdm(probe_xy, desc="Resolve .pth paths", total=len(probe_xy)):
        ex, ey = float(row[0]), float(row[1])
        for basename in _candidate_pth_basenames(pth_prefix, ex, ey):
            p = embedding_dir / basename
            sp = str(p)
            if sp in seen:
                continue
            if not p.is_file():
                continue
            if _parse_coords_from_pth_filename(p.name, search_pat) is None:
                continue
            seen.add(sp)
            paths.append(p)
    return paths


def _load_single_pth_for_dict(path, search_pat):
    p = Path(path)
    parsed = _parse_coords_from_pth_filename(p.name, search_pat)
    if parsed is None:
        return ("skip", None, None, None)
    key = (parsed[0], parsed[1])
    try:
        emb = _load_pth_embedding_file(p)
        return ("ok", key, emb, None)
    except Exception as e:
        return ("err", None, None, (p.name, str(e)))


def _list_pth_paths_capped(embedding_dir, cap=400_000):
    """
    List ``*.pth`` paths under ``embedding_dir``, unless there are at least ``cap`` files.

    Returns
    -------
    (paths, hit_cap)
        If ``hit_cap`` is True, ``paths`` is None (directory too large; use selective).
        Otherwise ``paths`` is the full list (length < cap).
    """
    embedding_dir = Path(embedding_dir)
    paths = []
    with os.scandir(embedding_dir) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".pth"):
                paths.append(embedding_dir / e.name)
                if len(paths) >= cap:
                    return None, True
    return paths, False


def load_embeddings_selective(
    celltype_df,
    embedding_dir,
    coord_cols=None,
    tolerance=1.0,
    pth_prefix="sc_NCRT",
    load_workers=0,
    selective_strategy="batch",
):
    """
    Load only .pth files needed for ``celltype_df`` rows (no ``glob('*.pth')``).

    - ``batch`` (default): unique patch lattice + parallel ``torch.load``.
    - ``per_cell``: legacy nested Python loop (very slow for millions of cells).

    When the directory mostly contains flat names like
    ``sc_NCRT_10000.018…_6157.051….pth``, a short preflight switches batch probes to
    **unique cell (x,y) only** (avoids ~10M+ lattice probes). Set
    ``ESCCAI_SELECTIVE_LEGACY_PROBES=1`` to force the old lattice+decimal union.
    ``ESCCAI_SELECTIVE_PREFLIGHT_MAX`` controls sample size (default 512).
    """
    embedding_dir = Path(embedding_dir)
    search_pat = _pth_coord_search_regex(pth_prefix)
    x_col, y_col = _resolve_cell_coord_columns(celltype_df, coord_cols)

    x_vals = celltype_df[x_col].to_numpy()
    y_vals = celltype_df[y_col].to_numpy()
    valid_mask = ~(pd.isna(x_vals) | pd.isna(y_vals))
    xv = x_vals[valid_mask].astype(np.float64, copy=False)
    yv = y_vals[valid_mask].astype(np.float64, copy=False)
    n_valid = int(xv.size)

    print(
        f"  Selective .pth loading: {n_valid:,} cells, tolerance={tolerance}, "
        f"prefix={pth_prefix!r}, strategy={selective_strategy!r}"
    )

    if selective_strategy == "per_cell":
        return _load_embeddings_selective_per_cell(
            embedding_dir,
            search_pat,
            pth_prefix,
            x_vals,
            y_vals,
            valid_mask,
            tolerance,
        )

    dbg_probes = os.environ.get("ESCCAI_DEBUG_EMBED_PROBES", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    raw_pf = os.environ.get("ESCCAI_SELECTIVE_PREFLIGHT_MAX", "").strip()
    try:
        preflight_n = int(raw_pf) if raw_pf else 512
    except ValueError:
        preflight_n = 512
    preflight_n = max(64, preflight_n)

    mode, mode_detail = _selective_embedding_probe_mode(
        embedding_dir, pth_prefix, search_pat, max_sample=preflight_n
    )
    if mode == "abort":
        print(
            f"  ⊘ Skipping selective basename resolution ({mode_detail}; scanned up to "
            f"{preflight_n:,} .pth). Flat ``{{prefix}}_x_y.pth`` probes would miss every "
            "file; caller will fall back to full-directory load."
        )
        return {}, []

    force_legacy = (
        os.environ.get("ESCCAI_SELECTIVE_LEGACY_PROBES", "").strip().lower()
        in ("1", "true", "yes")
    )
    if mode == "raw_unique" and not force_legacy:
        print(
            f"  Selective probe layout: unique cell coordinates only ({mode_detail}). "
            "Skipping lattice expansion (saves time for long-decimal flat basenames)."
        )
        probe_xy = _merge_batch_probe_coords_raw_unique(xv, yv, log_parts=dbg_probes)
        print(f"  Unique coordinate probes (raw cell x,y): {len(probe_xy):,}")
    else:
        if force_legacy and mode == "raw_unique":
            print("  ESCCAI_SELECTIVE_LEGACY_PROBES=1 — using lattice + rounded-decimal probes.")
        elif mode == "legacy":
            print(f"  Selective probe layout: legacy lattice + decimals ({mode_detail}).")
        probe_xy = _merge_batch_probe_coords(xv, yv, tolerance, log_parts=dbg_probes)
        print(f"  Unique coordinate probes (lattice + rounded cells): {len(probe_xy):,}")
        print(
            "  Note: probe count is not cell count — candidate (x,y) for path lookup; "
            "many cells share one patch, so loaded .pth count is often ≪ "
            f"{n_valid:,} rows."
        )

    paths = _collect_selective_pth_paths(embedding_dir, probe_xy, pth_prefix, search_pat)
    print(f"  Existing .pth paths matched: {len(paths):,}")

    from concurrent.futures import ThreadPoolExecutor

    workers = load_workers if load_workers and load_workers > 0 else min(
        16, max(4, (os.cpu_count() or 4))
    )
    embeddings_dict = {}
    failed_files = []

    if len(paths) == 0:
        return embeddings_dict, failed_files

    def _wrap_load_pth(p):
        return _load_single_pth_for_dict(p, search_pat)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, key, emb, err in tqdm(
            pool.map(
                _wrap_load_pth,
                paths,
                chunksize=max(1, len(paths) // (workers * 8)),
            ),
            total=len(paths),
            desc=f"Loading .pth (workers={workers})",
        ):
            if status == "ok" and key is not None:
                embeddings_dict[key] = emb
            elif status == "err" and err is not None:
                failed_files.append(err)
                if len(failed_files) <= 5:
                    print(f"  Warning: Failed to load {err[0]}: {err[1]}")

    print(f"  Unique .pth files read: {len(paths):,}")
    print(f"  Embedding coordinate keys: {len(embeddings_dict):,}")
    return embeddings_dict, failed_files


def _load_embeddings_selective_per_cell(
    embedding_dir,
    search_pat,
    pth_prefix,
    x_vals,
    y_vals,
    valid_mask,
    tolerance,
):
    embedding_dir = Path(embedding_dir)
    cell_indices = np.nonzero(valid_mask)[0]
    path_cache = {}
    missing_paths = set()
    embeddings_dict = {}
    failed_files = []
    n_valid = int(cell_indices.size)

    for i in tqdm(cell_indices, desc="Cells (resolve .pth)", total=n_valid):
        cx, cy = float(x_vals[i]), float(y_vals[i])
        found = False
        for ex, ey in _grid_coords_within_tolerance(cx, cy, tolerance):
            for basename in _candidate_pth_basenames(pth_prefix, ex, ey):
                p = embedding_dir / basename
                sp = str(p)
                if sp in missing_paths:
                    continue
                if not p.is_file():
                    missing_paths.add(sp)
                    continue
                parsed = _parse_coords_from_pth_filename(p.name, search_pat)
                if parsed is None:
                    missing_paths.add(sp)
                    continue
                key = (parsed[0], parsed[1])
                if sp in path_cache:
                    embeddings_dict[key] = path_cache[sp]
                    found = True
                    break
                try:
                    emb = _load_pth_embedding_file(p)
                    path_cache[sp] = emb
                    embeddings_dict[key] = emb
                    found = True
                    break
                except Exception as e:
                    failed_files.append((p.name, str(e)))
                    if len(failed_files) <= 5:
                        print(f"  Warning: Failed to load {p.name}: {e}")
            if found:
                break

    print(f"  Unique .pth files read: {len(path_cache):,}")
    print(f"  Embedding coordinate keys: {len(embeddings_dict):,}")
    return embeddings_dict, failed_files


def load_embeddings(pth_files, pth_prefix="sc_NCRT", load_workers=0):
    """
    Load all listed ``.pth`` files into a coordinate-keyed dict.

    ``load_workers``: parallel ``torch.load`` threads (0 = auto for large lists;
    1 = strictly sequential). Small lists skip threading overhead.
    """
    embeddings_dict = {}
    failed_files = []
    search_pat = _pth_coord_search_regex(pth_prefix)
    n_files = len(pth_files)
    if n_files == 0:
        return embeddings_dict, failed_files

    workers = (
        load_workers
        if load_workers and load_workers > 0
        else min(16, max(4, (os.cpu_count() or 4)))
    )
    use_parallel = workers > 1 and n_files >= 64

    print("  Loading embeddings...")
    if not use_parallel:
        for pth_file in tqdm(pth_files, desc="Loading .pth files"):
            parsed = _parse_coords_from_pth_filename(Path(pth_file).name, search_pat)
            if parsed is None:
                continue
            x_coord, y_coord = parsed
            try:
                embedding = _load_pth_embedding_file(pth_file)
                embeddings_dict[(x_coord, y_coord)] = embedding
            except Exception as e:
                failed_files.append((Path(pth_file).name, str(e)))
                if len(failed_files) <= 5:
                    print(f"  Warning: Failed to load {Path(pth_file).name}: {e}")
                continue
        return embeddings_dict, failed_files

    from concurrent.futures import ThreadPoolExecutor

    def _wrap(p):
        return _load_single_pth_for_dict(p, search_pat)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, key, emb, err in tqdm(
            pool.map(
                _wrap,
                pth_files,
                chunksize=max(1, n_files // (workers * 8)),
            ),
            total=n_files,
            desc=f"Loading .pth files (workers={workers})",
        ):
            if status == "ok" and key is not None:
                embeddings_dict[key] = emb
            elif status == "err" and err is not None:
                failed_files.append(err)
                if len(failed_files) <= 5:
                    print(f"  Warning: Failed to load {err[0]}: {err[1]}")

    return embeddings_dict, failed_files





##############################################################
# 2026.02.26 LLY Match embeddings with celltype data
##############################################################
import os
import numpy as np
from tqdm import tqdm

# -------------------------
# Step 1: Coordinate matching
# -------------------------
def find_matching_coord(target_x, target_y, coords_dict, tolerance=1.0):
    """Find matching coordinate in coords_dict within tolerance.
    
    Args:
        target_x, target_y: Target coordinates
        coords_dict: Dictionary with keys (x, y) and values embedding vectors
        tolerance: Maximum allowed distance for matching (default 1.0 pixel)
    
    Returns:
        Matching coordinate key (x, y) or None if no match found
    """
    # Use Euclidean distance for more accurate matching
    best_match = None
    best_distance = float('inf')
    
    for (x, y), emb in coords_dict.items():
        distance = ((x - target_x)**2 + (y - target_y)**2)**0.5
        if distance < tolerance and distance < best_distance:
            best_match = (x, y)
            best_distance = distance
    
    return best_match

# -------------------------
# Step 2: Match embeddings
# -------------------------
def match_embeddings(
    celltype_df,
    embeddings_dict,
    tolerance=1.0,
    coord_cols=None,
    group_col=None,
    level1_name=None,
):
    """Match celltype dataframe with embeddings based on coordinates.
    
    Args:
        celltype_df: DataFrame with coordinate columns
        embeddings_dict: dict with keys (x, y) and values embedding vectors
        tolerance: float, tolerance for coordinate matching
        coord_cols: tuple of (x_col, y_col) to specify which columns to use.
                   If None, auto-detect: priority is centroid_x/centroid_y, then X_pix_HE/Y_pix_HE
                   Examples: ('centroid_x', 'centroid_y') or ('X_pix_HE', 'Y_pix_HE')
        group_col: optional column name in ``celltype_df`` (e.g. ``'TumorID'``). When set, each
                   matched row includes ``'group'`` for Leave-One-Group-Out CV downstream.
        level1_name: optional column copied into each matched row for ``y_level1`` (e.g.
            ``'final_lineage'`` on Xenium CSVs that were not renamed to ``celltype_level1``).
    """
    matched_data = []
    
    x_col, y_col = _resolve_cell_coord_columns(celltype_df, coord_cols)
    if coord_cols is not None:
        print(f"  Using specified coordinates: {x_col}, {y_col}")
    elif x_col == "centroid_x":
        print(f"  Using coordinates: {x_col}, {y_col} (from StarDist matching)")
    else:
        print(f"  Using coordinates: {x_col}, {y_col}")

    # Speed-up: build a KDTree once, then do nearest-neighbor queries in batch.
    # This replaces the previous O(N_cells * N_embeddings) scan that called `find_matching_coord`
    # for every row.
    from scipy.spatial import cKDTree

    # Keep a stable index mapping between KDTree points and embeddings_dict values.
    embedding_keys = list(embeddings_dict.keys())
    if len(embedding_keys) == 0:
        return matched_data

    embedding_coords = np.asarray(embedding_keys, dtype=np.float32)  # shape (N, 2)
    embedding_vectors = [embeddings_dict[k] for k in embedding_keys]

    print(f"  Building KDTree for embeddings: {embedding_coords.shape[0]:,} points")
    tree = cKDTree(embedding_coords)

    # Extract cell coordinates in vector form.
    x_vals = celltype_df[x_col].to_numpy()
    y_vals = celltype_df[y_col].to_numpy()

    valid_mask = ~(pd.isna(x_vals) | pd.isna(y_vals))
    cell_indices = np.nonzero(valid_mask)[0]  # positions in dataframe (0..len-1)

    if cell_indices.size == 0:
        return matched_data

    cell_points = np.column_stack([x_vals[valid_mask], y_vals[valid_mask]]).astype(np.float32)

    # Query the nearest embedding for each cell point within `tolerance`.
    # For queries with no neighbor in range, cKDTree returns idx == tree.n.
    dists, nn_idxs = tree.query(cell_points, k=1, distance_upper_bound=tolerance)
    matched_mask = nn_idxs < len(embedding_vectors)
    matched_cell_indices = cell_indices[matched_mask]
    matched_nn_indices = nn_idxs[matched_mask].astype(int)

    if matched_cell_indices.size == 0:
        return matched_data

    if group_col is not None and group_col not in celltype_df.columns:
        raise ValueError(f"group_col={group_col!r} not found in celltype_df columns")

    # Gather arrays for faster post-processing.
    celltypes = celltype_df["celltype"].to_numpy()[matched_cell_indices]
    matched_x = x_vals[matched_cell_indices]
    matched_y = y_vals[matched_cell_indices]

    # Preserve original X_pix_HE/Y_pix_HE if they exist; otherwise fall back to matched coords.
    if "X_pix_HE" in celltype_df.columns:
        orig_x = celltype_df["X_pix_HE"].to_numpy()[matched_cell_indices]
    else:
        orig_x = matched_x
    if "Y_pix_HE" in celltype_df.columns:
        orig_y = celltype_df["Y_pix_HE"].to_numpy()[matched_cell_indices]
    else:
        orig_y = matched_y

    print(
        f"  KDTree matched {matched_cell_indices.size:,} / {len(celltype_df):,} cells "
        f"({matched_cell_indices.size / max(len(celltype_df), 1) * 100:.2f}%)."
    )

    hierarchy_cols = [
        c
        for c in celltype_df.columns
        if isinstance(c, str) and c.startswith("celltype_level")
    ]
    extra_label_cols = []
    if level1_name and level1_name in celltype_df.columns and level1_name not in hierarchy_cols:
        extra_label_cols.append(level1_name)

    # Build the same output format as before (list of dicts).
    for j, df_pos in enumerate(matched_cell_indices):
        srow = celltype_df.iloc[df_pos]
        row = {
            "idx": df_pos,
            "celltype": celltypes[j],
            "X_pix_HE": orig_x[j],
            "Y_pix_HE": orig_y[j],
            "centroid_x": matched_x[j],
            "centroid_y": matched_y[j],
            "embedding": embedding_vectors[matched_nn_indices[j]],
        }
        for c in hierarchy_cols + extra_label_cols:
            row[c] = srow[c]
        if "celltype_level1" not in row:
            row["celltype_level1"] = celltypes[j]
        if group_col is not None:
            row["group"] = srow[group_col]
        matched_data.append(row)

    return matched_data

# Extra hierarchy columns saved beside ``y_level1`` (from ``level1_name``).
_MATCHED_EXTRA_HIERARCHY = (
    ("celltype_level3", "y_level3"),
    ("celltype_level4", "y_level4"),
    ("celltype_level12", "y_level12"),
)


def _matched_rows_to_unicode_labels(matched_data, col):
    """One label per matched row as a unicode ndarray (NumPy 2-friendly, no object dtype)."""
    vals = []
    for item in matched_data:
        v = item.get(col, np.nan)
        if pd.isna(v):
            vals.append("")
        else:
            vals.append(str(v).strip())
    mxl = max((len(s) for s in vals), default=1)
    return np.asarray(vals, dtype=f"U{mxl}")


def _load_optional_unicode_y(loaded_data, key):
    if key not in loaded_data.files:
        return None
    return loaded_data[key]


def _npz_has_all_hierarchy_label_arrays(loaded_data):
    """True when cache contains L12 + L3 (CNiche) + L4 (TNiche) label arrays."""
    has_l12 = "y_level12" in loaded_data.files
    has_l3 = "y_level3" in loaded_data.files or "y_level0" in loaded_data.files
    has_l4 = "y_level4" in loaded_data.files or "y_level01" in loaded_data.files
    return has_l12 and has_l3 and has_l4


def _matched_npz_usable_as_cache(matched_features_path, rebuild_if_missing_hierarchy=True):
    """Return False when npz is missing or must be rebuilt (e.g. lacks hierarchy tiers)."""
    if not os.path.exists(matched_features_path):
        return False
    if not rebuild_if_missing_hierarchy:
        return True
    _peek = np.load(matched_features_path, allow_pickle=True)
    try:
        return _npz_has_all_hierarchy_label_arrays(_peek)
    finally:
        _peek.close()


def _print_matched_label_summary(y, y_level1, y_level3, y_level4, y_level12):
    """Print shapes / class counts for all tiers used in five-head training."""
    print(f"\nData shape:")
    print(f"  X alignment: {len(y)} rows")
    print(f"  y (L2 / celltype): {y.shape}, {len(np.unique(y))} classes")
    print(f"  y_level1 (lineage): {y_level1.shape}, {len(np.unique(y_level1))} classes")
    for name, arr in (
        ("y_level12 (sublineage)", y_level12),
        ("y_level3 (CNiche)", y_level3),
        ("y_level4 (TNiche)", y_level4),
    ):
        if arr is None:
            print(f"  {name}: MISSING (not in npz / not in CSV)")
        else:
            print(f"  {name}: {arr.shape}, {len(np.unique(arr))} classes")


# -------------------------
# Step 3: Main pipeline function
# -------------------------
def prepare_matched_embeddings(
    celltype_df,
    embeddings_dict,
    matched_features_path,
    tolerance=1.0,
    coord_cols=None,
    level1_name="celltype_level1",
    group_col=None,
    return_groups=False,
    rebuild_if_missing_hierarchy=True,
    require_hierarchy_labels=False,
):
    """
    Match embeddings with celltype dataframe, save to disk if not exist, and return X, y.
    
    Args:
        celltype_df: pd.DataFrame with columns ['X_pix_HE', 'Y_pix_HE', 'celltype'] or ['centroid_x', 'centroid_y', 'celltype']
        embeddings_dict: dict with keys (x, y) and values embedding vectors
        matched_features_path: str, path to save/load npz file
        tolerance: float, tolerance for coordinate matching
        coord_cols: tuple of (x_col, y_col) to specify which columns to use for matching.
                   If None, auto-detect: priority is centroid_x/centroid_y, then X_pix_HE/Y_pix_HE
                   Examples: ('centroid_x', 'centroid_y') or ('X_pix_HE', 'Y_pix_HE')
        level1_name: column in matched rows used for ``y_level1`` (e.g. ``'celltype_level1'``,
            ``'celltype_level12'``). Other fixed tiers are saved as ``y_level3``, ``y_level4``,
            ``y_level12`` when the CSV contains ``celltype_level3``, etc.
        group_col: optional; e.g. ``'TumorID'``. Per-row group ids are stored in the npz as
                   ``groups`` (aligned with ``X``) for Leave-One-Group-Out CV.
        return_groups: if True, append ``groups`` as the last return value (may be None).
        rebuild_if_missing_hierarchy: if True and the npz lacks ``y_level12`` / ``y_level3`` /
            ``y_level4``, ignore the cache and re-match + re-save (needed after enabling five-head).
        require_hierarchy_labels: if True, raise after load/save when any of L12/L3/L4 is missing.
        
    Returns:
        X, y, y_level1, y_level3, y_level4, y_level12, X_coords [, groups if return_groups].
        ``y_level3`` / ``y_level4`` / ``y_level12`` are None if absent from the npz (rebuild cache).
    """
    use_cache = _matched_npz_usable_as_cache(matched_features_path, rebuild_if_missing_hierarchy)
    if os.path.exists(matched_features_path) and not use_cache and rebuild_if_missing_hierarchy:
        print(
            "  ⚠ matched_features cache lacks y_level12 / y_level3 / y_level4. "
            "Rebuilding (set rebuild_if_missing_hierarchy=False to keep old cache)."
        )

    if use_cache:
        print("Loading pre-saved matched data...")
        # allow_pickle: older caches may store ``groups`` as dtype=object (strings).
        loaded_data = np.load(matched_features_path, allow_pickle=True)
        X = loaded_data['X']
        y = loaded_data['y']
        # Backward compatibility: old cache files may not contain y_level1.
        if 'y_level1' in loaded_data:
            y_level1 = loaded_data['y_level1']
        else:
            y_level1 = y
            print("  ⚠ y_level1 not found in saved data. Falling back to y (old cache format).")
        if "hierarchical_label_col" in loaded_data.files:
            cached_col = str(np.asarray(loaded_data["hierarchical_label_col"]).reshape(-1)[0])
            if cached_col != str(level1_name):
                print(
                    f"  ⚠ Cache hierarchical_label_col={cached_col!r} != level1_name={level1_name!r}. "
                    f"``y_level1`` still reflects the cached file; delete npz to rebuild."
                )
        y_level3 = _load_optional_unicode_y(loaded_data, "y_level3")
        if y_level3 is None:
            y_level3 = _load_optional_unicode_y(loaded_data, "y_level0")  # legacy npz
        y_level4 = _load_optional_unicode_y(loaded_data, "y_level4")
        if y_level4 is None:
            y_level4 = _load_optional_unicode_y(loaded_data, "y_level01")  # legacy npz
        y_level12 = _load_optional_unicode_y(loaded_data, "y_level12")
        print(f"  ✓ Loaded from {matched_features_path}")
        _print_matched_label_summary(y, y_level1, y_level3, y_level4, y_level12)
        if require_hierarchy_labels and (
            y_level12 is None or y_level3 is None or y_level4 is None
        ):
            raise ValueError(
                f"Hierarchy labels missing in {matched_features_path}. "
                "Delete the npz and re-run with XENIUM_CELL_COORD_COLUMN_RENAME_FULL, or pass "
                "rebuild_if_missing_hierarchy=True."
            )
        groups_loaded = loaded_data["groups"] if "groups" in loaded_data.files else None
        if group_col is not None and groups_loaded is None:
            print(
                f"  ⚠ Cache has no 'groups' but group_col={group_col!r} was set. "
                "Remove the npz to rebuild with groups, or load groups separately."
            )
        # print(f"  Classes: {sorted(np.unique(y))}")
        # Check if coordinates are saved
        if 'X_coords' in loaded_data:
            print(f"  ✓ Coordinates found in saved data")
            if return_groups:
                return (
                    X,
                    y,
                    y_level1,
                    y_level3,
                    y_level4,
                    y_level12,
                    loaded_data['X_coords'],
                    groups_loaded,
                )
            return X, y, y_level1, y_level3, y_level4, y_level12, loaded_data['X_coords']
        else:
            print(f"  ⚠ Coordinates not found in saved data (old format)")
            # Try to reconstruct coordinates from celltype_df
            # This requires matching by the order in which data was saved
            # For now, return None and let the caller handle it
            if return_groups:
                return X, y, y_level1, y_level3, y_level4, y_level12, None, groups_loaded
            return X, y, y_level1, y_level3, y_level4, y_level12, None

    _rename_map = celltype_df.attrs.get("_esccai_column_rename", {}) or {}
    _level1_col = resolve_label_column(level1_name, _rename_map)
    if _level1_col not in celltype_df.columns and level1_name in celltype_df.columns:
        _level1_col = level1_name

    print("Matching embeddings with celltype data (this may take a few minutes)...")
    matched_data = match_embeddings(
        celltype_df,
        embeddings_dict,
        tolerance,
        coord_cols=coord_cols,
        group_col=group_col,
        level1_name=_level1_col,
    )
    print(f"  Matched {len(matched_data)} cells out of {len(celltype_df)}")
    print(f"  Match rate: {len(matched_data)/len(celltype_df)*100:.2f}%")

    if len(matched_data) == 0:
        raise ValueError(
            "Matched 0 cells — embeddings dict may be empty (cache rebuild without loading .pth?). "
            "Delete the npz or fix match_hist2cell_matrix / reload embeddings before saving."
        )

    # Feature matrix and labels
    X = np.array([item['embedding'] for item in matched_data])
    y = np.array([item['celltype'] for item in matched_data])
    _level1_key = _level1_col if matched_data and _level1_col in matched_data[0] else level1_name
    if matched_data and _level1_key not in matched_data[0]:
        raise KeyError(
            f"level1_name={level1_name!r} (resolved {_level1_col!r}) not in matched rows. "
            f"Available keys: {sorted(matched_data[0].keys())!r}"
        )
    y_level1 = np.array([item[_level1_key] for item in matched_data])

    # Save matched data for future use (including coordinates for spatial visualization)
    print(f"\nSaving matched data for future use...")
    # Also save coordinates for spatial visualization (in the same order as X and y)
    X_coords = np.array([[item['X_pix_HE'], item['Y_pix_HE']] for item in matched_data])
    _hl = str(_level1_key)
    save_kw = dict(
        X=X,
        y=y,
        y_level1=y_level1,
        X_coords=X_coords,
        hierarchical_label_col=np.asarray(_hl, dtype=f"U{max(len(_hl), 1)}"),
    )
    _legacy_hier_col = {
        "celltype_level3": "celltype_level0",
        "celltype_level4": "celltype_level01",
    }
    for col, npy_key in _MATCHED_EXTRA_HIERARCHY:
        use_col = col
        if matched_data and use_col not in matched_data[0]:
            use_col = _legacy_hier_col.get(col, col)
        if matched_data and use_col in matched_data[0]:
            save_kw[npy_key] = _matched_rows_to_unicode_labels(matched_data, use_col)
            print(f"  ✓ Saving {npy_key} from column {use_col!r}")
    groups_arr = None
    if group_col is not None:
        # Unicode ndarray loads without pickle (NumPy 2 defaults allow_pickle=False).
        _glist = [str(item["group"]) for item in matched_data]
        _maxl = max((len(s) for s in _glist), default=1)
        groups_arr = np.asarray(_glist, dtype=f"U{_maxl}")
        save_kw["groups"] = groups_arr
        print(f"  ✓ Including groups column {group_col!r} in npz ({len(np.unique(groups_arr))} unique)")
    np.savez_compressed(matched_features_path, **save_kw)
    print(f"  ✓ Saved features to {matched_features_path}")
    print(f"  ✓ Saved coordinates for spatial visualization")

    y_level3 = save_kw.get("y_level3")
    y_level4 = save_kw.get("y_level4")
    y_level12 = save_kw.get("y_level12")
    _print_matched_label_summary(y, y_level1, y_level3, y_level4, y_level12)
    if matched_data:
        print(f"  Matched row keys (sample): {sorted(matched_data[0].keys())}")

    if require_hierarchy_labels and (
        y_level12 is None or y_level3 is None or y_level4 is None
    ):
        _need = ["celltype_level12", "celltype_level3", "celltype_level4"]
        _have = [c for c in celltype_df.columns if str(c).startswith("celltype_level")]
        raise ValueError(
            "Could not save y_level12 / y_level3 / y_level4 to npz. "
            f"CSV columns after rename should include {_need}; "
            f"found hierarchy columns: {_have}. "
            "Use column_rename=base.XENIUM_CELL_COORD_COLUMN_RENAME_FULL."
        )

    if return_groups:
        return X, y, y_level1, y_level3, y_level4, y_level12, X_coords, groups_arr
    return X, y, y_level1, y_level3, y_level4, y_level12, X_coords



## Evaluation function
from sklearn.metrics import f1_score, accuracy_score, classification_report

def evaluate(model, loader, device, scaler=None, spatial_ctx=None):
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        raise TypeError(
            "evaluate() expects `model` to be a torch.nn.Module, got "
            f"{type(model).__module__}.{type(model).__name__}. "
            "Avoid `import model` in notebooks (it shadows the trained model name `model`); "
            "use `import model as model_pkg` or pass `LP['model']` explicitly."
        )
    model.eval()
    all_preds = []
    all_labels = []
    use_spatial = (
        spatial_ctx is not None
        and getattr(model, "use_spatial_context", False)
        and spatial_ctx.get("X_all") is not None
    )
    if use_spatial:
        X_all = spatial_ctx["X_all"].to(device, non_blocking=True)
        neighbor_index = spatial_ctx["neighbor_index"].to(device, non_blocking=True)

    with torch.no_grad():
        for batch in loader:
            global_idx = None
            if use_spatial and len(batch) >= 3:
                x, y = batch[0], batch[1]
                global_idx = batch[-1]
            else:
                x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            neighbor_x = None
            if use_spatial and global_idx is not None:
                global_idx = global_idx.to(device, non_blocking=True)
                neighbor_x = gather_neighbor_embeddings(
                    X_all, neighbor_index, global_idx, device=device
                )

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(x, neighbor_x=neighbor_x)
            else:
                logits = model(x, neighbor_x=neighbor_x)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    return acc, macro_f1, weighted_f1, all_preds, all_labels




######################################
# 2026.02.26 LLY Evaluate and plot on all data
######################################
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def evaluate_and_plot_on_all_data(
    model,
    matched_features_path,
    class_names,
    evaluate,
    plot_celltype_spatial_distribution,
    CellTypeDataset,
    device,
    scaler,
    val_loader=None,
    X=None,
    y=None,
    y_encoded=None,
    X_train=None,
    X_train_scaled=None,
    X_test_scaled=None,
    y_train_encoded=None,
    y_test_encoded=None,
    min_samples_per_class=29,
    X_coords_matched=None,
    celltype_pixel_NCRT_tumor1=None,
    celltype_pred_dir=None,
    celltype_true_dir=None,
    prediction_only=False,  # If True, skip loading labels to predict all samples
    spatial_figsize=(12, 10),
    spatial_point_size=0.6,
    spatial_plot_mode="pred_true_l2",
    spatial_color_scheme="ncrt",
    class_names_level1=None,
    y_level1_f=None,
    y_level1_encoded_f=None,
    celltype_pred_level1_dir=None,
    celltype_true_level1_dir=None,
    spatial_title_pred_l2=None,
    spatial_title_true_l2=None,
    spatial_title_pred_l1=None,
    spatial_title_true_l1=None,
    neighbor_index=None,
):
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        raise TypeError(
            "evaluate_and_plot_on_all_data: `model` must be a torch.nn.Module, got "
            f"{type(model).__module__}.{type(model).__name__}. "
            "Notebooks often shadow this name with `import model`; use `import model as model_pkg` "
            "or pass the trained network explicitly (e.g. ``LP['model']``)."
        )

    spatial_plot_mode = str(spatial_plot_mode or "pred_true_l2").lower()
    if spatial_plot_mode not in ("pred_true_l2", "pred_l2_pred_l1", "pred_l2_only"):
        raise ValueError(
            "spatial_plot_mode must be one of: pred_true_l2, pred_l2_pred_l1, pred_l2_only"
        )
    plot_l2_true = spatial_plot_mode == "pred_true_l2"
    plot_l1_pred = spatial_plot_mode == "pred_l2_pred_l1" and class_names_level1 is not None

    use_spatial_infer = (
        neighbor_index is not None and getattr(model, "use_spatial_context", False)
    )
    if use_spatial_infer:
        print(f"  Spatial inference: enabled (neighbor_index shape={np.asarray(neighbor_index).shape})")

    def _neighbor_x_for_batch(X_full_t, nbr_idx_t, row_offset, x_batch):
        if not use_spatial_infer or X_full_t is None:
            return None
        b = x_batch.shape[0]
        global_idx = torch.arange(row_offset, row_offset + b, dtype=torch.long, device=device)
        return gather_neighbor_embeddings(X_full_t, nbr_idx_t, global_idx, device=device)

    def _collect_dual_head_preds(loader, X_full_t=None, nbr_idx_t=None, row_start=0):
        from tqdm import tqdm

        model.eval()
        preds_l2, preds_l1 = [], []
        has_head = hasattr(model, "forward_heads") and callable(getattr(model, "forward_heads"))
        row_offset = int(row_start)
        with torch.no_grad():
            for x, _ in tqdm(loader, desc="Predicting (L2+L1 heads)"):
                x = x.to(device, non_blocking=True)
                neighbor_x = _neighbor_x_for_batch(X_full_t, nbr_idx_t, row_offset, x)
                row_offset += x.shape[0]
                if has_head:
                    if torch.cuda.is_available():
                        with torch.amp.autocast("cuda"):
                            logits_l2, logits_l1 = forward_heads_l2_l1(model, x, neighbor_x=neighbor_x)
                    else:
                        logits_l2, logits_l1 = forward_heads_l2_l1(model, x, neighbor_x=neighbor_x)
                    preds_l2.append(torch.argmax(logits_l2, dim=1).cpu().numpy())
                    preds_l1.append(torch.argmax(logits_l1, dim=1).cpu().numpy())
                else:
                    logits = model(x, neighbor_x=neighbor_x)
                    preds_l2.append(torch.argmax(logits, dim=1).cpu().numpy())
        out_l2 = np.concatenate(preds_l2) if preds_l2 else np.array([], dtype=np.int64)
        out_l1 = np.concatenate(preds_l1) if preds_l1 else None
        return out_l2, out_l1

    # Priority 1: Load data directly from matched_features_path (for new predictions)
    X_all_scaled = None
    y_all_encoded = None
    has_labels = False
    y_original = None  # Store original labels for visualization (when prediction_only=True)
    
    if matched_features_path and os.path.exists(matched_features_path):
        print("  Loading data from matched_features_path...")
        try:
            loaded_data = load_matched_features_bundle(matched_features_path)
            X_loaded = loaded_data['X']
            print(f"  ✓ Loaded X from {matched_features_path}: {X_loaded.shape}")
            
            # Check if labels exist
            # If prediction_only=True, still load labels for visualization but skip evaluation
            if 'y' in loaded_data:
                y_loaded = loaded_data['y']
                # Store original labels for visualization (as strings if possible)
                y_original = y_loaded.copy() if hasattr(y_loaded, 'copy') else y_loaded
                
                if prediction_only:
                    print(f"  ✓ Loaded y from {matched_features_path}: {y_loaded.shape}, dtype: {y_loaded.dtype}")
                    print(f"  ⚠ prediction_only=True: Labels will be used for visualization only, not for evaluation.")
                    has_labels = False  # Don't evaluate, but keep labels for visualization
                else:
                    has_labels = True
                    print(f"  ✓ Loaded y from {matched_features_path}: {y_loaded.shape}, dtype: {y_loaded.dtype}")
                    print(f"  ⚠ Note: Labels will be used for evaluation. If encoding mismatch occurs, set prediction_only=True.")
                
                # Check if y is string type - keep as string for visualization
                is_string = False
                if y_loaded.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object (string)
                    is_string = True
                elif len(y_loaded) > 0 and isinstance(y_loaded[0], (str, np.str_)):
                    is_string = True
                
                if is_string:
                    # Store string labels for visualization
                    y_string_labels = y_loaded.copy() if hasattr(y_loaded, 'copy') else y_loaded
                    
                    if not prediction_only:
                        # Only encode if we need to evaluate
                        print("  ⚠ y contains string labels. Encoding to integers...")
                        print("  ⚠ Warning: New LabelEncoder may produce different encoding than training!")
                        print("  ⚠ This can cause label mismatch if data has different classes than training.")
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_all_encoded = le.fit_transform(y_loaded)
                        n_unique = len(np.unique(y_all_encoded))
                        print(f"  ✓ Encoded y to integers: {y_all_encoded.shape}, dtype: {y_all_encoded.dtype}")
                        print(f"  ⚠ Encoded labels have {n_unique} unique classes (0-{n_unique-1})")
                        print(f"  ⚠ Model expects {len(class_names) if 'class_names' in globals() else 'unknown'} classes")
                    else:
                        # For prediction_only, don't encode - will use string labels directly for visualization
                        y_all_encoded = None
                        print(f"  ✓ Keeping labels as strings for visualization (no encoding needed)")
                else:
                    # y is already numeric
                    if not prediction_only:
                        y_all_encoded = y_loaded.astype(np.int64)
                        n_unique = len(np.unique(y_all_encoded))
                        print(f"  ✓ y is already numeric, converted to int64")
                        print(f"  ⚠ Labels have {n_unique} unique classes (0-{n_unique-1})")
                        print(f"  ⚠ Model expects {len(class_names) if 'class_names' in globals() else 'unknown'} classes")
                    else:
                        y_all_encoded = None
                        print(f"  ✓ Labels are numeric, will use for visualization")
            else:
                print("  ⚠ No labels (y) found in matched_features_path. Will only predict, not evaluate.")
                y_original = None
            
            # Scale the features using the provided scaler
            if isinstance(scaler, StandardScaler):
                X_all_scaled = scaler.transform(X_loaded)
                print("  ✓ Scaled X using provided scaler")
            else:
                print("  ⚠ Warning: scaler is not StandardScaler. Attempting to use as-is...")
                X_all_scaled = X_loaded
            
            if not has_labels:
                y_all_encoded = None
        except Exception as e:
            print(f"  ⚠ Failed to load from matched_features_path: {e}")
    
    # Priority 2: Use provided X and y
    if X_all_scaled is None:
        if X is not None:
            print("  Using provided X and y...")
            if isinstance(scaler, StandardScaler):
                X_all_scaled = scaler.transform(X)
            else:
                print("  ⚠ Warning: scaler is not StandardScaler, recreating from X_train...")
                if X_train is None:
                    raise ValueError("X_train not found. Cannot recreate StandardScaler.")
                feature_scaler = StandardScaler()
                feature_scaler.fit(X_train)
                X_all_scaled = feature_scaler.transform(X)
                print("  ✓ Recreated StandardScaler from X_train")
            
            if y_encoded is not None:
                y_all_encoded = y_encoded
                has_labels = True
                if y is not None:
                    y_original = np.asarray(y)
                    print("  ✓ Using provided y strings as original labels for spatial TRUE plot.")
            elif y is not None:
                y_original = np.asarray(y)
                if prediction_only:
                    has_labels = False
                    print(
                        "  ✓ prediction_only=True: y kept as original strings "
                        "(not re-encoded with ad-hoc LabelEncoder)."
                    )
                else:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_all_encoded = le.fit_transform(y)
                    has_labels = True
        # Priority 3: Reconstruct from train/test split (fallback)
        elif X_train_scaled is not None and X_test_scaled is not None:
            print("  Warning: Reconstructing from train/test split...")
            print("  Note: Order may not match coordinates due to train_test_split shuffling.")
            X_all_scaled = np.vstack([X_train_scaled, X_test_scaled])
            if y_train_encoded is not None and y_test_encoded is not None:
                y_all_encoded = np.concatenate([y_train_encoded, y_test_encoded])
                has_labels = True
        else:
            raise ValueError("No data source found. Please provide X, matched_features_path, or X_train_scaled/X_test_scaled.")

    # Validate data shapes before creating dataset
    print(f"  X_all_scaled shape: {X_all_scaled.shape}")
    if has_labels:
        print(f"  y_all_encoded shape: {y_all_encoded.shape}")
        if len(X_all_scaled) != len(y_all_encoded):
            min_len = min(len(X_all_scaled), len(y_all_encoded))
            print(f"  ⚠ Warning: X ({len(X_all_scaled)}) and y ({len(y_all_encoded)}) length mismatch. Truncating to {min_len}")
            print(f"  ⚠ This will result in fewer predictions than coordinates. Consider checking data consistency.")
            X_all_scaled = X_all_scaled[:min_len]
            y_all_encoded = y_all_encoded[:min_len]
    else:
        print("  No labels available - prediction mode only")

    # Create DataLoader for all data
    # If no labels, create dummy labels (will be ignored during prediction)
    if has_labels:
        # Ensure y_all_encoded is integer type
        if y_all_encoded.dtype.kind in ['U', 'S', 'O']:  # Still string type
            print("  ⚠ Warning: y_all_encoded is still string type. Encoding to integers...")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_all_encoded = le.fit_transform(y_all_encoded)
        # Convert to int64 to ensure compatibility with torch.LongTensor
        y_all_encoded = y_all_encoded.astype(np.int64)
        print(f"  ✓ Final y_all_encoded dtype: {y_all_encoded.dtype}")
        all_dataset = CellTypeDataset(X_all_scaled, y_all_encoded)

        print("X_all_scaled: ", X_all_scaled.shape)
        print("y_all_encoded: ", y_all_encoded.shape)
    else:
        # Create dummy labels for dataset (won't be used)
        dummy_labels = np.zeros(len(X_all_scaled), dtype=np.int64)
        all_dataset = CellTypeDataset(X_all_scaled, dummy_labels)

        print("X_all_scaled: ", X_all_scaled.shape)
        print("dummy_labels: ", dummy_labels.shape)

    if val_loader is not None:
        batch_size = val_loader.batch_size
        num_workers = val_loader.num_workers
        pin_memory = val_loader.pin_memory
    else:
        # Default values if val_loader is not provided
        batch_size = 1024 if torch.cuda.is_available() else 256
        num_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = True if torch.cuda.is_available() else False
    
    # For large datasets, reduce num_workers to avoid indexing issues
    # Set num_workers=0 if dataset is large to avoid multiprocessing issues
    if len(all_dataset) > 50000:
        if num_workers > 0:
            print(f"  ⚠ Large dataset detected ({len(all_dataset)} samples). Setting num_workers=0 to avoid indexing issues.")
            num_workers = 0
    
    all_loader = DataLoader(
        all_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory if num_workers == 0 else False  # Disable pin_memory when num_workers=0
    )
    print(f"  Total samples: {len(all_dataset)}")
    print(f"  DataLoader batch_size: {batch_size}, num_workers: {num_workers}")

    X_full_t = nbr_idx_t = spatial_ctx = None
    if use_spatial_infer:
        nbr_idx_t = torch.as_tensor(np.asarray(neighbor_index), dtype=torch.long)
        X_full_t = torch.as_tensor(np.asarray(X_all_scaled, dtype=np.float32))
        spatial_ctx = {"X_all": X_full_t, "neighbor_index": nbr_idx_t}
        gidx = torch.arange(len(X_all_scaled), dtype=torch.long)
        if has_labels:
            all_dataset = TensorDataset(
                torch.from_numpy(np.asarray(X_all_scaled, dtype=np.float32)),
                torch.from_numpy(np.asarray(y_all_encoded, dtype=np.int64)),
                gidx,
            )
        else:
            all_dataset = TensorDataset(
                torch.from_numpy(np.asarray(X_all_scaled, dtype=np.float32)),
                torch.zeros(len(X_all_scaled), dtype=torch.long),
                gidx,
            )
        all_loader = DataLoader(
            all_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory if num_workers == 0 else False,
        )

    all_preds_l1 = None

    def _spatial_scheme_for_tier(tier):
        sch = (spatial_color_scheme or "ncrt").lower()
        if sch in ("ncrt", "xenium_lineage", "xenium_ct"):
            return sch
        if sch in ("xenium", "xenium_auto"):
            return "xenium_lineage" if tier == "l1" else "xenium_ct"
        return sch

    # Predict on all data
    if has_labels:
        # Evaluate on all data (with labels)
        all_acc, all_macro_f1, all_weighted_f1, all_preds, all_labels = evaluate(
            model, all_loader, device, scaler, spatial_ctx=spatial_ctx
        )
        # Keep raw arrays for spatial plotting to preserve 1:1 order with coordinates.
        plot_preds = np.asarray(all_preds).copy()
        plot_labels = np.asarray(all_labels).copy() if all_labels is not None else None
        # Keep original string labels for true-celltype plotting when available.
        raw_true_celltypes = y_original.copy() if y_original is not None else None
        print(f"\nFinal Performance on All Data:")
        print(f"  Accuracy: {all_acc:.4f}")
        print(f"  Macro F1-score: {all_macro_f1:.4f}")
        print(f"  Weighted F1-score: {all_weighted_f1:.4f}")
        print("all_preds: ", all_preds.shape)

        # Get class names for present classes
        # Filter out indices that are out of bounds for class_names
        max_class_idx = len(class_names) - 1
        unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
        print("unique_labels: ", unique_labels.shape)
        
        # Check for out-of-bounds indices (for evaluation report only).
        out_of_bounds_labels = unique_labels[unique_labels > max_class_idx]
        if len(out_of_bounds_labels) > 0:
            print(f"  ⚠ Warning: Found {len(out_of_bounds_labels)} label(s) out of bounds: {out_of_bounds_labels}")
            print(f"  ⚠ class_names has {len(class_names)} classes (indices 0-{max_class_idx})")
            print(f"  ⚠ This suggests label encoding mismatch between training and prediction data.")
            print(f"  ⚠ Filtering out out-of-bounds labels/predictions...")
            
            # Filter predictions and labels to valid range (keep only samples where both are valid)
            valid_pred_mask = all_preds <= max_class_idx
            valid_label_mask = all_labels <= max_class_idx
            valid_mask = valid_pred_mask & valid_label_mask
            
            if not np.all(valid_mask):
                n_filtered = np.sum(~valid_mask)
                n_invalid_preds = np.sum(~valid_pred_mask)
                n_invalid_labels = np.sum(~valid_label_mask)
                print(f"  ⚠ Filtered out {n_filtered} samples ({n_filtered/len(all_preds)*100:.1f}%) with out-of-bounds indices")
                if n_invalid_preds > 0:
                    print(f"    - {n_invalid_preds} predictions with out-of-bounds indices")
                if n_invalid_labels > 0:
                    print(f"    - {n_invalid_labels} labels with out-of-bounds indices")
                    print(f"    - Note: Labels were re-encoded during loading, which may not match training encoding.")
                    print(f"    - Suggestion: Use training LabelEncoder or predict without labels for full dataset.")
                
                # Only filter if we have a significant number of valid samples
                if np.sum(valid_mask) < len(all_preds) * 0.5:
                    print(f"  ⚠ Warning: Less than 50% of samples are valid. Consider checking label encoding.")
                
                eval_preds = all_preds[valid_mask]
                eval_labels = all_labels[valid_mask]
            else:
                eval_preds = all_preds
                eval_labels = all_labels
        else:
            eval_preds = all_preds
            eval_labels = all_labels
        # Recalculate unique labels for evaluation report
        unique_labels = np.unique(np.concatenate([eval_labels, eval_preds]))
        
        # Only use labels that are within class_names range
        valid_unique_labels = unique_labels[unique_labels <= max_class_idx]
        actual_class_names = [class_names[i] for i in valid_unique_labels]

        print(f"\nDetailed Classification Report:")
        print(f"  Number of classes: {len(valid_unique_labels)}")
        print(classification_report(eval_labels, eval_preds, labels=valid_unique_labels, target_names=actual_class_names, zero_division=0))

        # Convert predictions/labels to class names for spatial plotting (raw order).
        all_pred_celltypes = np.array([class_names[pred] if pred <= max_class_idx else f"Unknown_{pred}" for pred in plot_preds])
        if raw_true_celltypes is not None:
            if len(raw_true_celltypes) == len(plot_preds):
                all_true_celltypes = np.asarray(raw_true_celltypes)
                print("  ✓ Using original labels for TRUE spatial plot (avoids encoding mismatch).")
            else:
                all_true_celltypes = np.asarray(raw_true_celltypes)[:len(plot_preds)]
                print("  ⚠ True labels length mismatch; truncated original labels to match predictions.")
        else:
            all_true_celltypes = np.array([class_names[label] if label <= max_class_idx else f"Unknown_{label}" for label in plot_labels])

        # Keep raw predictions/labels for downstream spatial coordinate alignment.
        all_preds = plot_preds
        all_labels = plot_labels
        if plot_l1_pred:
            _, all_preds_l1 = _collect_dual_head_preds(all_loader, X_full_t, nbr_idx_t, row_start=0)
    else:
        # Prediction mode only (no labels for evaluation, but may have labels for visualization)
        print("  Predicting on new data (no labels for evaluation)...")
        if plot_l1_pred:
            all_preds, all_preds_l1 = _collect_dual_head_preds(all_loader, X_full_t, nbr_idx_t, row_start=0)
        else:
            model.eval()
            all_preds = []
            row_offset = 0

            with torch.no_grad():
                for x, _ in tqdm(all_loader, desc="Predicting"):
                    x = x.to(device, non_blocking=True)
                    neighbor_x = _neighbor_x_for_batch(X_full_t, nbr_idx_t, row_offset, x)
                    row_offset += x.shape[0]

                    if torch.cuda.is_available():
                        with torch.amp.autocast("cuda"):
                            logits = model(x, neighbor_x=neighbor_x)
                    else:
                        logits = model(x, neighbor_x=neighbor_x)

                    preds = torch.argmax(logits, dim=1)
                    all_preds.append(preds.cpu().numpy())

            all_preds = np.concatenate(all_preds)
        all_labels = None  # Optional: will be filled if true labels can be aligned.
        all_acc = None
        all_macro_f1 = None
        all_weighted_f1 = None
        
        print(f"\nPrediction completed:")
        print(f"  Total predictions: {len(all_preds)}")
        print(f"  Predicted classes: {np.unique(all_preds)}")
        
        # Check for out-of-bounds predictions
        max_class_idx = len(class_names) - 1
        out_of_bounds_preds = all_preds[all_preds > max_class_idx]
        if len(out_of_bounds_preds) > 0:
            print(f"  ⚠ Warning: Found {len(out_of_bounds_preds)} prediction(s) out of bounds: {np.unique(out_of_bounds_preds)}")
            print(f"  ⚠ class_names has {len(class_names)} classes (indices 0-{max_class_idx})")
            print(f"  ⚠ These predictions will be labeled as 'Unknown_<index>'")
        
        # Convert predictions to class names (with bounds checking)
        all_pred_celltypes = np.array([class_names[pred] if pred <= max_class_idx else f"Unknown_{pred}" for pred in all_preds])
        
        # Check if we have original labels for visualization (from prediction_only mode)
        if y_original is not None:
            # Use original string labels directly for visualization
            if len(y_original) == len(all_preds):
                all_true_celltypes = y_original
                print(f"  ✓ Using original labels for visualization: {len(all_true_celltypes)} labels")
                # Try to align labels to class indices so accuracy/F1 can be computed and
                # downstream bar plots (e.g. plot_per_class_accuracy) can be drawn.
                try:
                    y_arr = np.asarray(y_original)
                    if y_arr.dtype.kind in ['U', 'S', 'O']:
                        mapped = encode_labels_with_class_names(y_arr, class_names)
                        valid_mask = mapped >= 0
                        if not np.all(valid_mask):
                            n_bad = int((~valid_mask).sum())
                            print(
                                f"  ⚠ {n_bad} true labels are not in class_names; "
                                "excluding them from metric computation."
                            )
                        if np.any(valid_mask):
                            eval_labels = mapped[valid_mask]
                            eval_preds = all_preds[valid_mask]
                            all_labels = mapped
                            from sklearn.metrics import accuracy_score, f1_score

                            all_acc = accuracy_score(eval_labels, eval_preds)
                            all_macro_f1 = f1_score(eval_labels, eval_preds, average="macro", zero_division=0)
                            all_weighted_f1 = f1_score(eval_labels, eval_preds, average="weighted", zero_division=0)
                            print(
                                "  ✓ Computed metrics in prediction_only mode from aligned true labels: "
                                f"acc={all_acc:.4f}, macro_f1={all_macro_f1:.4f}, weighted_f1={all_weighted_f1:.4f}"
                            )
                    else:
                        labels_num = y_arr.astype(np.int64)
                        all_labels = labels_num
                        from sklearn.metrics import accuracy_score, f1_score

                        all_acc = accuracy_score(labels_num, all_preds)
                        all_macro_f1 = f1_score(labels_num, all_preds, average="macro", zero_division=0)
                        all_weighted_f1 = f1_score(labels_num, all_preds, average="weighted", zero_division=0)
                        print(
                            "  ✓ Computed metrics in prediction_only mode from numeric true labels: "
                            f"acc={all_acc:.4f}, macro_f1={all_macro_f1:.4f}, weighted_f1={all_weighted_f1:.4f}"
                        )
                except Exception as e:
                    print(f"  ⚠ Could not compute metrics from true labels in prediction_only mode: {e}")
            else:
                print(f"  ⚠ Warning: Original labels length ({len(y_original)}) doesn't match predictions ({len(all_preds)})")
                # Truncate or pad to match
                if len(y_original) > len(all_preds):
                    all_true_celltypes = y_original[:len(all_preds)]
                    print(f"  ✓ Truncated labels to match predictions")
                else:
                    all_true_celltypes = None
        else:
            all_true_celltypes = None  # No true labels available

    print(f"\nPreparing spatial distribution plot...")

    # Match predictions with coordinates for spatial visualization
    print("  Matching predictions with coordinates...")
    X_coords = None

    # Priority 1: Load from saved npz (most reliable, matches the order of X and y)
    if os.path.exists(matched_features_path):
        try:
            loaded_data = load_matched_features_bundle(matched_features_path)
            if 'X_coords' in loaded_data:
                X_coords = loaded_data['X_coords']
                print(f"  ✓ Loaded coordinates from {matched_features_path}: {X_coords.shape}")
                if len(X_coords) != len(all_preds):
                    print(f"  ⚠ Coordinate count ({len(X_coords)}) doesn't match predictions ({len(all_preds)})")
                    print(f"  This may happen if data was split into train/test. Using available coordinates.")
                    # If coordinates are fewer, we can still use them for the matched subset
                    if len(X_coords) < len(all_preds):
                        print(f"  ⚠ Warning: Only {len(X_coords)} coordinates available for {len(all_preds)} predictions")
                        print(f"  Will plot only the first {len(X_coords)} predictions with coordinates")
                        X_coords = None  # Will be handled later
                    else:
                        # If more coordinates, truncate to match
                        X_coords = X_coords[:len(all_preds)]
                        print(f"  ✓ Truncated coordinates to match predictions: {X_coords.shape}")
        except Exception as e:
            print(f"  ⚠ Failed to load coordinates from npz: {e}")

    # Priority 2: Coordinates from embedding step (if not loaded from npz)
    if X_coords is None and X_coords_matched is not None:
        X_coords = X_coords_matched
        print(f"  ✓ Using coordinates from prepare_matched_embeddings: {X_coords.shape}")
        if len(X_coords) != len(all_preds):
            print(f"  ⚠ Coordinate count ({len(X_coords)}) doesn't match predictions ({len(all_preds)})")
            if len(X_coords) < len(all_preds):
                print(f"  ⚠ Warning: Only {len(X_coords)} coordinates available for {len(all_preds)} predictions")
                print(f"  Will plot only the first {len(X_coords)} predictions with coordinates")
            else:
                # If more coordinates, truncate to match
                X_coords = X_coords[:len(all_preds)]
                print(f"  ✓ Truncated coordinates to match predictions: {X_coords.shape}")

    # Priority 3: Reconstruct from celltype data if needed
    if X_coords is None or len(X_coords) != len(all_preds):
        print("  Reconstructing coordinates from celltype data with same filters...")
        # Load celltype_pixel_NCRT_tumor1 if not provided
        if celltype_pixel_NCRT_tumor1 is None:
            # Extract base path from matched_features_path
            # matched_features_path format: /path/to/data/NCRT/matched_features_tumor1.npz
            # We need: /path/to/data/
            base_path = os.path.dirname(os.path.dirname(matched_features_path))
            tumor1_coords_path = os.path.join(base_path, 'he_cell_coords', 'NCRT_CellPixCoords_tumor1.csv')
            if os.path.exists(tumor1_coords_path):
                celltype_pixel_NCRT_tumor1 = pd.read_csv(tumor1_coords_path)
                print(f"  Loaded celltype_pixel_NCRT_tumor1 from {tumor1_coords_path}")
            else:
                raise ValueError("celltype_pixel_NCRT_tumor1 not found. Please load it first.")

        # Step 1: Filter out Low_quality and non_specific
        celltype_after_quality_filter = celltype_pixel_NCRT_tumor1[
            ~celltype_pixel_NCRT_tumor1['celltype'].isin(['Low_quality', 'non_specific'])
        ].copy().reset_index(drop=True)

        # Step 2: Apply gap-based filtering
        class_counts = celltype_after_quality_filter['celltype'].value_counts()
        print("class_counts: ", class_counts)
        threshold = min_samples_per_class
        valid_classes = class_counts[class_counts >= threshold].index
        celltype_filtered = celltype_after_quality_filter[
            celltype_after_quality_filter['celltype'].isin(valid_classes)
        ].copy().reset_index(drop=True)

        # Extract coordinates
        X_coords = celltype_filtered[['X_pix_HE', 'Y_pix_HE']].values
        print(f"  ✓ Reconstructed coordinates from filtered celltype data")
        print("X_coords: ", X_coords.shape)
    # Check if shapes match
    if X_coords is not None:
        print(f"  Coordinates shape: {X_coords.shape}")
        print(f"  Predictions shape: {all_preds.shape}")
        
        # Handle case where coordinates don't match predictions
        if len(X_coords) != len(all_preds):
            if len(X_coords) < len(all_preds):
                print(f"  ⚠ Truncating predictions to match available coordinates")
                all_preds = all_preds[:len(X_coords)]
                all_pred_celltypes = all_pred_celltypes[:len(X_coords)]
                if all_labels is not None:
                    all_labels = all_labels[:len(X_coords)]
                if all_true_celltypes is not None:
                    all_true_celltypes = all_true_celltypes[:len(X_coords)]
                print(f"  ✓ Adjusted predictions shape: {all_preds.shape}")
            else:
                # Truncate coordinates
                X_coords = X_coords[:len(all_preds)]
                print(f"  ✓ Truncated coordinates to match predictions: {X_coords.shape}")

    if X_coords is not None and len(X_coords) == len(all_preds):
        pred_df_dict = {
            'X_pix_HE': X_coords[:, 0],
            'Y_pix_HE': X_coords[:, 1],
            # 'predicted_celltype': all_pred_celltypes
            'celltype': all_pred_celltypes

        }
        # Only add true_celltype if labels are available
        if all_true_celltypes is not None:
            pred_df_dict['true_celltype'] = all_true_celltypes
        pred_df = pd.DataFrame(pred_df_dict)
        print(f"  ✓ Created prediction DataFrame with {len(pred_df)} cells")

        l2_scheme = _spatial_scheme_for_tier("l2")
        l2_color_overrides = None
        if l2_scheme != "ncrt":
            try:
                import sys
                from pathlib import Path as _Path

                _xdir = _Path(__file__).resolve().parent.parent / "Xenium_lung"
                if _xdir.is_dir() and str(_xdir) not in sys.path:
                    sys.path.insert(0, str(_xdir))
                from plot_HEanno_spatial_labels import (
                    resolve_xenium_spatial_color_overrides,
                )

                tier = "ct" if l2_scheme == "xenium_ct" else "auto"
                l2_color_overrides = resolve_xenium_spatial_color_overrides(
                    class_names,
                    tier=tier,
                    canonical_labels=class_names,
                )
            except Exception as exc:
                print(f"  ⚠ Could not build shared L2 spatial colors: {exc}")
        print(f"\nPlotting spatial distribution of PREDICTED level2 cell types...")
        plot_celltype_spatial_distribution(
            pred_df,
            x_col='X_pix_HE',
            y_col='Y_pix_HE',
            celltype_col='celltype',
            figsize=spatial_figsize,
            alpha=0.6,
            s=spatial_point_size,
            format='jpg',
            save_path=celltype_pred_dir,
            color_overrides=l2_color_overrides,
            spatial_color_scheme=l2_scheme,
            title=spatial_title_pred_l2 or "Predicted level2 spatial distribution",
        )

        if plot_l2_true and all_true_celltypes is not None:
            print(f"\nPlotting spatial distribution of TRUE level2 cell types...")
            plot_celltype_spatial_distribution(
                pred_df,
                x_col='X_pix_HE',   
                y_col='Y_pix_HE',
                celltype_col='true_celltype',
                figsize=spatial_figsize,
                alpha=0.6,
                s=spatial_point_size,
                format='jpg',
                save_path=celltype_true_dir,
                color_overrides=l2_color_overrides,
                spatial_color_scheme=l2_scheme,
                title=spatial_title_true_l2 or "True level2 spatial distribution",
            )
        elif all_true_celltypes is not None and not plot_l2_true:
            print(
                "  Skipping duplicate level2 TRUE spatial plot "
                f"(spatial_plot_mode={spatial_plot_mode!r})."
            )

        if plot_l1_pred and all_preds_l1 is not None:
            n_l1 = len(class_names_level1)
            valid_l1 = (all_preds_l1 >= 0) & (all_preds_l1 < n_l1)
            if not np.all(valid_l1):
                print(
                    f"  ⚠ Dropping {(~valid_l1).sum()} cells with out-of-range level1 head predictions."
                )
            idx_l1 = np.where(valid_l1)[0]
            pred_l1_names = np.array(
                [class_names_level1[int(i)] for i in all_preds_l1[valid_l1]]
            )
            pred_df_l1 = pd.DataFrame({
                "X_pix_HE": X_coords[idx_l1, 0],
                "Y_pix_HE": X_coords[idx_l1, 1],
                "celltype": pred_l1_names,
            })
            if y_level1_f is not None and len(y_level1_f) == len(all_preds):
                pred_df_l1["true_celltype"] = np.asarray(y_level1_f)[idx_l1]
            print(f"\nPlotting spatial distribution of PREDICTED level1 (lineage)...")
            plot_celltype_spatial_distribution(
                pred_df_l1,
                x_col="X_pix_HE",
                y_col="Y_pix_HE",
                celltype_col="celltype",
                figsize=spatial_figsize,
                alpha=0.6,
                s=spatial_point_size,
                format="jpg",
                save_path=celltype_pred_level1_dir,
                spatial_color_scheme=_spatial_scheme_for_tier("l1"),
                title=spatial_title_pred_l1 or "Predicted level1 spatial distribution",
            )
            if (
                celltype_true_level1_dir is not None
                and "true_celltype" in pred_df_l1.columns
            ):
                plot_celltype_spatial_distribution(
                    pred_df_l1,
                    x_col="X_pix_HE",
                    y_col="Y_pix_HE",
                    celltype_col="true_celltype",
                    figsize=spatial_figsize,
                    alpha=0.6,
                    s=spatial_point_size,
                    format="jpg",
                    save_path=celltype_true_level1_dir,
                    spatial_color_scheme=_spatial_scheme_for_tier("l1"),
                    title=spatial_title_true_l1 or "True level1 spatial distribution",
                )
        elif plot_l1_pred:
            print(
                "  ⚠ spatial_plot_mode=pred_l2_pred_l1 but model has no forward_heads(); "
                "skip level1 spatial plot."
            )
    else:
        if X_coords is None:
            print(f"  ⚠ Warning: No coordinates available for spatial visualization!")
            print(f"    Predictions shape: {all_preds.shape}")
            print(f"    This may be due to coordinate loading failure or data mismatch.")
        else:
            print(f"  ⚠ Warning: Shape mismatch!")
            print(f"    Coordinates: {X_coords.shape}")
            print(f"    Predictions: {all_preds.shape}")
            print(f"    This may be due to additional filtering. Please check the data pipeline.")

    return all_acc, all_macro_f1, all_weighted_f1, all_preds, all_labels


def export_all_data_prediction_confidence(
    model,
    X,
    device,
    class_names,
    class_names_level1,
    y_encoded_f,
    y_level1_encoded_f,
    *,
    scaler=None,
    X_coords=None,
    y_true_l2=None,
    y_true_l1=None,
    csv_path=None,
    level2_hist_path=None,
    level1_hist_path=None,
    batch_size=1024,
):
    """Export full-data level2/level1 prediction confidence and histogram plots."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_np = np.asarray(X)
    if scaler is not None:
        X_scaled = scaler.transform(X_np)
    else:
        X_scaled = X_np

    # Build fixed L2 -> L1 mapping from the filtered label space used by training.
    num_l2 = len(class_names)
    num_l1 = len(class_names_level1)
    child_to_parent = np.full(num_l2, -1, dtype=np.int64)
    for l2, l1 in zip(np.asarray(y_encoded_f), np.asarray(y_level1_encoded_f)):
        if child_to_parent[int(l2)] == -1:
            child_to_parent[int(l2)] = int(l1)
        elif child_to_parent[int(l2)] != int(l1):
            raise ValueError(f"Inconsistent hierarchy for L2 class {l2}")
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing L1 mapping for L2 classes: {missing}")

    map_l2_to_l1 = torch.zeros(num_l2, num_l1, dtype=torch.float32, device=device)
    for l2_idx, l1_idx in enumerate(child_to_parent):
        map_l2_to_l1[l2_idx, int(l1_idx)] = 1.0

    dataset = TensorDataset(torch.as_tensor(X_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs_l2_all = []
    probs_l1_head_all = []
    probs_l1_agg_all = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    if hasattr(model, "forward_heads"):
                        logits_l2, logits_l1_head = forward_heads_l2_l1(model, xb)
                    else:
                        logits_l2 = model(xb)
                        logits_l1_head = None
            else:
                if hasattr(model, "forward_heads"):
                    logits_l2, logits_l1_head = forward_heads_l2_l1(model, xb)
                else:
                    logits_l2 = model(xb)
                    logits_l1_head = None
            probs_l2 = torch.softmax(logits_l2.float(), dim=1)
            probs_l1_agg = torch.matmul(probs_l2, map_l2_to_l1)
            if logits_l1_head is not None:
                probs_l1_head = torch.softmax(logits_l1_head.float(), dim=1)
            else:
                probs_l1_head = probs_l1_agg
            probs_l2_all.append(probs_l2.cpu().numpy())
            probs_l1_head_all.append(probs_l1_head.cpu().numpy())
            probs_l1_agg_all.append(probs_l1_agg.cpu().numpy())

    probs_l2_all = np.concatenate(probs_l2_all, axis=0)
    probs_l1_head_all = np.concatenate(probs_l1_head_all, axis=0)
    probs_l1_agg_all = np.concatenate(probs_l1_agg_all, axis=0)
    pred_l2 = np.argmax(probs_l2_all, axis=1)
    conf_l2 = np.max(probs_l2_all, axis=1)
    pred_l1 = np.argmax(probs_l1_head_all, axis=1)
    conf_l1 = np.max(probs_l1_head_all, axis=1)
    pred_l1_agg = np.argmax(probs_l1_agg_all, axis=1)
    conf_l1_agg = np.max(probs_l1_agg_all, axis=1)

    export_df = pd.DataFrame(
        {
            "pred_level2_idx": pred_l2,
            "pred_level2": [class_names[i] for i in pred_l2],
            "conf_level2": conf_l2,
            "pred_level1_idx": pred_l1,
            "pred_level1": [class_names_level1[i] for i in pred_l1],
            "conf_level1": conf_l1,
            "pred_level1_agg_idx": pred_l1_agg,
            "pred_level1_agg": [class_names_level1[i] for i in pred_l1_agg],
            "conf_level1_agg": conf_l1_agg,
        }
    )
    if y_true_l2 is not None:
        y_true_l2 = np.asarray(y_true_l2)
        export_df["true_level2_idx"] = y_true_l2
        valid_l2 = (y_true_l2 >= 0) & (y_true_l2 < len(class_names))
        export_df["true_level2"] = np.where(
            valid_l2, np.asarray(class_names, dtype=object)[y_true_l2.clip(0)], "Unknown"
        )
    if y_true_l1 is not None:
        y_true_l1 = np.asarray(y_true_l1)
        export_df["true_level1_idx"] = y_true_l1
        valid_l1 = (y_true_l1 >= 0) & (y_true_l1 < len(class_names_level1))
        export_df["true_level1"] = np.where(
            valid_l1,
            np.asarray(class_names_level1, dtype=object)[y_true_l1.clip(0)],
            "Unknown",
        )
    if X_coords is not None:
        X_coords = np.asarray(X_coords)
        if len(X_coords) == len(export_df):
            export_df["coord_x"] = X_coords[:, 0]
            export_df["coord_y"] = X_coords[:, 1]

    if csv_path:
        export_df.to_csv(csv_path, index=False)
        print(f"Saved all-data prediction confidence CSV to {csv_path}")

    def _plot_conf_hist(values, title, save_path):
        if not save_path:
            return
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=50, color="steelblue", edgecolor="black", alpha=0.85)
        plt.title(title)
        plt.xlabel("Confidence")
        plt.ylabel("Cell count")
        plt.xlim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved confidence histogram to {save_path}")

    _plot_conf_hist(
        conf_l2,
        "All-data level2 prediction confidence",
        level2_hist_path,
    )
    _plot_conf_hist(
        conf_l1,
        "All-data level1-head prediction confidence",
        level1_hist_path,
    )

    return export_df



##################################################
# 2026.03.03 LLY Match StarCoords to ROI
##################################################
import numpy as np
import pandas as pd

def match_StarCoords2ROI(star_coords, roi_coords, therapy_data):
    """
    Match star_coords (cell coordinates) to ROI rectangles and
    add a 'TumorID' column like 'NCRT_tumor1', 'NCRT_tumor2', etc.
    Also print and return the cell count for each ROI.

    Parameters
    ----------
    star_coords : pd.DataFrame
        Must contain columns: ['centroid_x', 'centroid_y', 'probability', ...]
    roi_coords : pd.DataFrame
        Must contain columns:
        ['Name', 'TopLeft_X', 'TopLeft_Y', 'TopRight_X', 'TopRight_Y',
         'BottomRight_X', 'BottomRight_Y', 'BottomLeft_X', 'BottomLeft_Y']
    therapy_data : str
        Prefix for TumorID, e.g. 'NCRT'

    Returns
    -------
    star_coords : pd.DataFrame
        star_coords with 'TumorID' column, NaN TumorID rows removed.
    star_coords_in_roi : pd.DataFrame
        Same as star_coords (after filtering), kept for convenience.
    roi_cell_counts : pd.DataFrame
        Columns: ['TumorID', 'cell_count'], sorted by TumorID numeric order.
    """

    # Remove duplicate coordinates (optional)
    star_coords = star_coords.drop_duplicates(
        subset=['centroid_x', 'centroid_y'],
        keep='first'
    )

    # Add a new column 'TumorID' at the first column position, initialized with NaN
    star_coords.insert(0, 'TumorID', np.nan)

    # Iterate over each ROI and assign TumorID to points within the ROI
    for _, roi in roi_coords.iterrows():
        # e.g. "NCRT_tumor1"
        name = f"{therapy_data}_{roi['Name']}"

        # For a rectangle, X range and Y range are defined by min/max of the corners
        min_x = min(roi['TopLeft_X'],  roi['BottomLeft_X'])
        max_x = max(roi['TopRight_X'], roi['BottomRight_X'])
        min_y = min(roi['TopLeft_Y'],  roi['TopRight_Y'])
        max_y = max(roi['BottomLeft_Y'], roi['BottomRight_Y'])

        # Boolean mask: points that fall inside this ROI
        mask = (
            (star_coords['centroid_x'] >= min_x) &
            (star_coords['centroid_x'] <= max_x) &
            (star_coords['centroid_y'] >= min_y) &
            (star_coords['centroid_y'] <= max_y)
        )

        # Assign TumorID (e.g. "NCRT_tumor2") to those points
        star_coords.loc[mask, 'TumorID'] = name

    # 1. Count how many rows have NaN TumorID
    nan_count = star_coords['TumorID'].isna().sum()
    print(f"Number of rows with TumorID = NaN: {nan_count}")

    # 2. Drop rows where TumorID is NaN
    star_coords.dropna(subset=['TumorID'], inplace=True)

    # 3. Reset index (optional, for clean 0..N-1 indexing)
    star_coords.reset_index(drop=True, inplace=True)

    print("star_coords shape after filtering:", star_coords.shape)
    print(star_coords.head())

    # Keep only points that lie in any ROI
    star_coords_in_roi = star_coords[star_coords['TumorID'].notna()].reset_index(drop=True)

    # ---- Count cells per ROI and sort by TumorID (ascending) ----
    # value_counts gives counts per TumorID
    roi_cell_counts = (
        star_coords['TumorID']
        .value_counts()                 # Series: index=TumorID, value=count
        .rename('cell_count')
        .reset_index()                  # columns: ['index', 'cell_count']
        .rename(columns={'index': 'TumorID'})
    )

    # Extract numeric part from TumorID (e.g. 'NCRT_tumor28' -> 28) for proper numeric sorting
    roi_cell_counts['tumor_num'] = roi_cell_counts['TumorID'].str.extract(r'(\d+)').astype(int)
    roi_cell_counts = roi_cell_counts.sort_values('tumor_num').drop(columns='tumor_num')

    print("Nuclei number within each ROI (sorted by TumorID):")
    print(roi_cell_counts)

    return star_coords_in_roi, roi_cell_counts
    # return star_coords, star_coords_in_roi, roi_cell_counts


########################################################
# 2026.03.04 LLY Match celltype to stardist
######################################################
import numpy as np
from scipy.spatial import cKDTree

def match_celltype2stardist(celltype_pixel_coords,
                            stardist_pixel_coords,
                            k_neighbors=100,
                            celltype_pixel_coords_cols=None,
                            stardist_pixel_coords_cols=None,
                            output_x_col=None,
                            output_y_col=None):
    """
    Match celltype_pixel_coords points to stardist_pixel_coords using KDTree
    and a greedy 1-to-1 nearest‑neighbor strategy.

    Parameters
    ----------
    celltype_pixel_coords : pd.DataFrame
        DataFrame containing cell type data with coordinate columns.
    stardist_pixel_coords : pd.DataFrame
        DataFrame containing StarDist coordinates with coordinate columns.
    k_neighbors : int, optional
        Number of nearest neighbors to search for each celltype point
        (upper bound; the effective value is min(k_neighbors, n_star)).
        Default: 100
    celltype_pixel_coords_cols : tuple or list of str, optional
        Column names for x and y coordinates in celltype_pixel_coords.
        Default: ('X_pix_HE', 'Y_pix_HE')
    stardist_pixel_coords_cols : tuple or list of str, optional
        Column names for x and y coordinates in stardist_pixel_coords.
        Default: ('centroid_x', 'centroid_y')
    output_x_col : str, optional
        Column name for the matched x coordinate in the output DataFrame.
        If None, uses the first column name from stardist_pixel_coords_cols.
        Default: None (uses stardist_pixel_coords_cols[0])
    output_y_col : str, optional
        Column name for the matched y coordinate in the output DataFrame.
        If None, uses the second column name from stardist_pixel_coords_cols.
        Default: None (uses stardist_pixel_coords_cols[1])

    Returns
    -------
    celltype_matched : pd.DataFrame
        Copy of `celltype_pixel_coords` with extra columns:
        - output_x_col      : matched x coordinate from stardist_pixel_coords
        - output_y_col      : matched y coordinate from stardist_pixel_coords
        - 'matched_distance': distance to the matched centroid
    """
    # Reset index so NumPy position indices always align with dataframe assignment.
    # Upstream filtering often keeps sparse original labels, which breaks `.loc` later.
    # celltype_pixel_coords = celltype_pixel_coords.reset_index(drop=True).copy()
    # stardist_pixel_coords = stardist_pixel_coords.reset_index(drop=True).copy()
    celltype_pixel_coords = celltype_pixel_coords.reset_index(drop=True).copy()
    stardist_pixel_coords = stardist_pixel_coords.reset_index(drop=True).copy()

    # Set default coordinate column names
    if celltype_pixel_coords_cols is None:
        celltype_pixel_coords_cols = ('X_pix_HE', 'Y_pix_HE')
    if stardist_pixel_coords_cols is None:
        stardist_pixel_coords_cols = ('centroid_x', 'centroid_y')
    
    # Convert to list if tuple
    if isinstance(celltype_pixel_coords_cols, tuple):
        celltype_pixel_coords_cols = list(celltype_pixel_coords_cols)
    if isinstance(stardist_pixel_coords_cols, tuple):
        stardist_pixel_coords_cols = list(stardist_pixel_coords_cols)
    
    # Set default output column names from stardist_pixel_coords_cols if not provided
    # output_x_col defaults to the first column of stardist_pixel_coords_cols
    if output_x_col is None:
        output_x_col = stardist_pixel_coords_cols[0]
    # output_y_col defaults to the second column of stardist_pixel_coords_cols
    if output_y_col is None:
        output_y_col = stardist_pixel_coords_cols[1]
    
    # Validate column names exist
    missing_celltype_cols = [col for col in celltype_pixel_coords_cols if col not in celltype_pixel_coords.columns]
    if missing_celltype_cols:
        raise ValueError(f"Missing columns in celltype_pixel_coords: {missing_celltype_cols}")
    
    missing_star_cols = [col for col in stardist_pixel_coords_cols if col not in stardist_pixel_coords.columns]
    if missing_star_cols:
        raise ValueError(f"Missing columns in stardist_pixel_coords: {missing_star_cols}")
    
    # Extract coordinates as numpy arrays
    celltype_coords = celltype_pixel_coords[celltype_pixel_coords_cols].values
    star_coords = stardist_pixel_coords[stardist_pixel_coords_cols].values
    print(f"celltype_pixel_coords shape: {celltype_coords.shape}")
    print(f"stardist_pixel_coords shape: {star_coords.shape}")

    if celltype_coords.shape[0] == 0:
        raise ValueError(
            "celltype_pixel_coords has 0 rows (empty GroundTruth after TumorID filter?). "
            "TumorID in GroundTruth must match therapy_data (e.g. SA_tumor3)."
        )
    if star_coords.shape[0] == 0:
        raise ValueError(
            "stardist_pixel_coords has 0 rows (no StarDist nuclei in the selected ROI / TumorID)."
        )

    # Build KDTree for star_coords (typically the larger set)
    print("Building KDTree for star_coords...")
    star_tree = cKDTree(star_coords)

    # Number of neighbors to query for each celltype point
    k_neighbors = min(int(k_neighbors), len(star_coords))
    if k_neighbors < 1:
        raise ValueError("Not enough star_coords points to perform matching.")

    print(f"Finding {k_neighbors} nearest neighbors for each celltype point...")
    distances, indices = star_tree.query(celltype_coords, k=k_neighbors)
    # Ensure 2D shape when k_neighbors == 1
    if k_neighbors == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]
    print(f"Nearest neighbor search completed. Distance matrix shape: {distances.shape}")

    # Greedy 1‑to‑1 matching: each celltype point is matched to the
    # nearest star point that has not been used yet.
    print("\nUsing greedy nearest neighbor matching...")

    matched_star_indices = set()
    celltype_matched_indices = []
    star_matched_indices = []
    matched_distances_list = []

    # Distance of closest neighbor for each celltype point
    nearest_distances = distances[:, 0]

    # Process celltype points from closest to farthest
    sorted_celltype_indices = np.argsort(nearest_distances)

    for celltype_idx in sorted_celltype_indices:
        # For this celltype point, iterate over its k nearest star points
        for neighbor_rank in range(k_neighbors):
            star_idx = int(indices[celltype_idx, neighbor_rank])
            if star_idx not in matched_star_indices:
                matched_star_indices.add(star_idx)
                celltype_matched_indices.append(celltype_idx)
                star_matched_indices.append(star_idx)
                matched_distances_list.append(distances[celltype_idx, neighbor_rank])
                break

    celltype_matched_indices = np.array(celltype_matched_indices, dtype=int)
    star_matched_indices = np.array(star_matched_indices, dtype=int)
    matched_distances_array = np.array(matched_distances_list, dtype=float)

    print(f"Matched {len(celltype_matched_indices)} points")
    if len(matched_distances_array) > 0:
        print(
            "Matched distance: Mean, Median, Max: "
            f"{matched_distances_array.mean():.2f}, "
            f"{np.median(matched_distances_array):.2f}, "
            f"{matched_distances_array.max():.2f}"
        )
    else:
        print("Matched distance: no pairs (empty); check inputs and TumorID overlap.")

    # Create a copy of the input DF and add matched information
    celltype_matched = celltype_pixel_coords.copy()
    celltype_matched[output_x_col] = np.nan
    celltype_matched[output_y_col] = np.nan
    celltype_matched['matched_distance'] = np.nan

    # Fill in matched coordinates/distances with position-based indexing
    # celltype_matched.loc[celltype_matched_indices, output_x_col] = star_coords[star_matched_indices, 0]
    # celltype_matched.loc[celltype_matched_indices, output_y_col] = star_coords[star_matched_indices, 1]
    # celltype_matched.loc[celltype_matched_indices, 'matched_distance'] = matched_distances_array
    x_col_idx = celltype_matched.columns.get_loc(output_x_col)
    y_col_idx = celltype_matched.columns.get_loc(output_y_col)
    d_col_idx = celltype_matched.columns.get_loc('matched_distance')
    celltype_matched.iloc[celltype_matched_indices, x_col_idx] = star_coords[star_matched_indices, 0]
    celltype_matched.iloc[celltype_matched_indices, y_col_idx] = star_coords[star_matched_indices, 1]
    celltype_matched.iloc[celltype_matched_indices, d_col_idx] = matched_distances_array

    print(f"\nMatched points summary:")
    print(f"  Total points in celltype_pixel_coords: {len(celltype_pixel_coords)}")
    print(f"  Matched points: {len(celltype_matched_indices)}")
    print(f"  Unmatched points: {len(celltype_pixel_coords) - len(celltype_matched_indices)}")

    # check the NAN
    celltype_matched_valid = celltype_matched.dropna(subset=[output_x_col, output_y_col])
    print(f"Matched of PCF-HE-StarDist: {celltype_matched_valid.shape}")

    return celltype_matched_valid
    # return celltype_matched, celltype_matched_indices, star_matched_indices, matched_distances_array



######################################################
# 2026.03.13 LLY Merge celltype level1 and level2
######################################################
def merge_celltype_level12(level1_path, level2_path, save_path, run=True):

    if run:
        ## run and save celltype_anno_df if run is True
        celltype_anno_df_level1 = pd.read_csv(level1_path)   # dont contain 'low_quality' and 'non_specific' cell types
        celltype_anno_df_level2 = pd.read_csv(level2_path)    
        level1_meta_cols = [col for col in celltype_anno_df_level1.columns[:-1] if col != 'celltype']
        print(f"level1_meta_cols: {level1_meta_cols}")
        level1_merge_cols = ['celltype'] + level1_meta_cols
        print(f"level1_merge_cols: {level1_merge_cols}")
        celltype_anno_df = celltype_anno_df_level2.merge(
            celltype_anno_df_level1[level1_merge_cols],
            on='celltype',
            how='left'
        )
        print("The number of cells:", celltype_anno_df.shape[0])
        if 'celltype_level1' in celltype_anno_df.columns:
            celltype_anno_df = celltype_anno_df[celltype_anno_df['celltype_level1'].notna()]
        # Align TumorID with StarDist/ROI (SA_*): codex exports may use S_tumor* / A_tumor*.
        if "TumorID" in celltype_anno_df.columns:
            tid = celltype_anno_df["TumorID"].astype(str)
            tid_norm = tid.str.replace(r"^A_", "SA_", regex=True).str.replace(r"^S_", "SA_", regex=True)
            if not tid_norm.equals(tid):
                celltype_anno_df = celltype_anno_df.copy()
                celltype_anno_df["TumorID"] = tid_norm
                print("[merge_celltype_level12] TumorID normalized: A_* / S_* -> SA_* before save")
        celltype_anno_df.to_csv(save_path, index=False)
        print(f"Celltype annotation DataFrame saved to {save_path}")
    else:
        ## load celltype_anno_df if run is False
        celltype_anno_df = pd.read_csv(save_path)

    print("The number of cells after deleting 'Low_quality' and 'non_specific' cells:", celltype_anno_df.shape[0])
    print("The number of cell types level2:", len(celltype_anno_df['celltype'].unique()))
    print("The number of cell types level1:", len(celltype_anno_df['celltype_level1'].unique()))

    return celltype_anno_df


######################################################
# 2026.03.18 LLY Make PCF2HE alignment in one function
######################################################
def make_PCF2HE_alignment(celltype_anno_df, celltype_coords_df, 
                          therapy_data, segment_path, 
                          segment_batch1='1', segment_batch2='2', 
                          value='NCRT', parent_value=None, 
                          save_path=None, save_path4ViT=None):

    if celltype_anno_df['celltype_level1'].isna().any():
        print("Warning: celltype_level1 contains NaN values!")

    ## QuPath shows HE->PCF, we need PCF->HE (INVERT the matrix)
    PCF2HE_transformation_NCRT = transformation_matrix_properties(therapy_data)

    ## Verifying PCF -> HE alignment
    # fig_verify = verify_transformation_alignment(he_path, pcf_path, PCF2HE_transformation_NCRT, 
    #                                              test_regions=1, region_size=4000)
    pixel_size_um_PCF = get_pixel_size(segment_path, segment_number=segment_batch1)    # segment_number=1 is PCF
    pixel_size_um_HE = get_pixel_size(segment_path, segment_number=segment_batch2)    # segment_number=2 is HE

    ## celltype_NCRT
    celltype_NCRT_measure = get_celltype_coords(celltype=celltype_anno_df, ncrt_anno_df=celltype_coords_df, 
                                                 value=value, parent_value=parent_value)
    celltype_pixel_NCRT_measure = add_pixel_coords(
        celltype_NCRT_measure,
        pixel_size_um_PCF,
        PCF2HE_transformation_NCRT,
        pixel_size_um_HE=pixel_size_um_HE,
        sanity_check=True,
    )

    ## Check NAN in celltype_pixel_NCRT_measure
    print("Number of cells with NAN in X_pix_HE:", celltype_pixel_NCRT_measure[celltype_pixel_NCRT_measure['X_pix_HE'].isna()].shape[0])
    print("Number of cells with NAN in Y_pix_HE:", celltype_pixel_NCRT_measure[celltype_pixel_NCRT_measure['Y_pix_HE'].isna()].shape[0])
    print("Number of cells with NAN in X_pix_HE and Y_pix_HE:", celltype_pixel_NCRT_measure[celltype_pixel_NCRT_measure['X_pix_HE'].isna() & celltype_pixel_NCRT_measure['Y_pix_HE'].isna()].shape[0])

    print("Cell numbers with NCRT:", celltype_pixel_NCRT_measure.shape[0])
    print(celltype_pixel_NCRT_measure.shape)
    # print(celltype_pixel_NCRT_measure.head())
    if save_path is not None:
        celltype_pixel_NCRT_measure.to_csv(save_path, index=False)
        print(f"Celltype pixel coordinates DataFrame saved to {save_path}")
    if save_path4ViT is not None:
        celltype_pixel_NCRT_measure4ViT = celltype_pixel_NCRT_measure.copy()
        celltype_pixel_NCRT_measure4ViT.rename(columns={'X_pix_HE': 'pxl_row_in_fullres', 'Y_pix_HE': 'pxl_col_in_fullres'}, inplace=True)
        celltype_pixel_NCRT_measure4ViT.to_csv(save_path4ViT, index=False)
        print(f"Celltype pixel coordinates DataFrame for ViT saved to {save_path4ViT}")
    else:
        print(f"Celltype pixel coordinates DataFrame not saved.")

    return celltype_pixel_NCRT_measure


######################################################
# 2026.03.19 For model input: LLY Load celltype pixel coordinates
######################################################
# Canonical label columns used by match_embeddings / prepare_matched_embeddings.
REQUIRED_CELL_LABEL_COL = "celltype"

# Applied when ``celltype`` is missing and ``column_rename`` is None (Xenium Select4 CSVs).
DEFAULT_CELL_COORD_COLUMN_RENAME_XENIUM = {
    "final_CT": "celltype",
    "final_sublineage": "celltype_level12",
}

# Optional full Xenium → NCRT-style hierarchy (pass explicitly via ``column_rename``).
XENIUM_CELL_COORD_COLUMN_RENAME_FULL = {
    **DEFAULT_CELL_COORD_COLUMN_RENAME_XENIUM,
    "final_lineage": "celltype_level1",
    "CNiche": "celltype_level3",
    "TNiche": "celltype_level4",
}

# Xenium training: coarse ``y`` (4 lineages) + fine ``y_level1`` (~33–45 final_CT types).
# Use with ``level1_name="celltype_level1"`` and ``auto_rename=False``.
XENIUM_COLUMN_RENAME_LINEAGE_Y_CT_LEVEL1 = {
    "final_lineage": "celltype_level1",
    "final_CT": "celltype",
}


def default_codex_meta_level012_path():
    """Path to ``data/codex_meta_celltype_level012.csv`` under the esccAI repo root."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(pkg_dir, "..", ".."))
    return os.path.join(repo_root, "data", "codex_meta_celltype_level012.csv")


def load_codex_meta_hierarchy_df(meta_path=None):
    """Load fixed celltype → level12 → level1 (and level0/01) mapping table."""
    meta_path = meta_path or default_codex_meta_level012_path()
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"codex meta hierarchy CSV not found: {meta_path}")
    df = pd.read_csv(meta_path)
    required = {
        "celltype",
        "celltype_level12",
        "celltype_level1",
        "celltype_level0",
        "celltype_level01",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"codex meta missing columns: {sorted(missing)}")
    return df


def build_l2_l12_l1_maps_from_codex_meta(class_names_l2, meta_path=None):
    """
    Build fixed index maps from codex meta for each level2 class name in ``class_names_l2``.

    Returns
    -------
    dict with keys:
        child_l2_to_l12, child_l2_to_l1, child_l12_to_l1 (np.int64 arrays aligned to L2 indices),
        meta (DataFrame), unmatched_l2 (list of class names missing from meta).
    """
    meta = load_codex_meta_hierarchy_df(meta_path)
    l2_to_l12 = {}
    l2_to_l1 = {}
    l12_to_l1 = {}
    for _, row in meta.iterrows():
        ct = str(row["celltype"]).strip()
        l12 = str(row["celltype_level12"]).strip()
        l1 = str(row["celltype_level1"]).strip()
        l2_to_l12[ct] = l12
        l2_to_l1[ct] = l1
        l12_to_l1[l12] = l1

    class_names_l2 = list(class_names_l2)
    child_l2_to_l12 = np.full(len(class_names_l2), -1, dtype=np.int64)
    child_l2_to_l1 = np.full(len(class_names_l2), -1, dtype=np.int64)
    unmatched = []
    l12_to_idx = {n: i for i, n in enumerate(sorted(set(l2_to_l12.values())))}
    l1_to_idx = {n: i for i, n in enumerate(sorted(set(l2_to_l1.values())))}

    for i, name in enumerate(class_names_l2):
        key = str(name).strip()
        if key not in l2_to_l12:
            unmatched.append(key)
            continue
        l12_name = l2_to_l12[key]
        l1_name = l2_to_l1[key]
        if l12_name in l12_to_idx:
            child_l2_to_l12[i] = l12_to_idx[l12_name]
        if l1_name in l1_to_idx:
            child_l2_to_l1[i] = l1_to_idx[l1_name]

    child_l12_to_l1 = np.full(len(l12_to_idx), -1, dtype=np.int64)
    for l12_name, l1_name in l12_to_l1.items():
        if l12_name in l12_to_idx and l1_name in l1_to_idx:
            child_l12_to_l1[l12_to_idx[l12_name]] = l1_to_idx[l1_name]

    return {
        "child_l2_to_l12": child_l2_to_l12,
        "child_l2_to_l1_meta": child_l2_to_l1,
        "child_l12_to_l1": child_l12_to_l1,
        "meta": meta,
        "unmatched_l2": unmatched,
        "class_names_level12_meta": list(l12_to_idx.keys()),
        "class_names_level1_meta": list(l1_to_idx.keys()),
    }


def _encode_optional_tier_after_mask(y_arr, mask):
    """Label-encode optional tier labels on the same boolean mask as L2 filtering."""
    if y_arr is None:
        return None, None, None
    y_arr = np.asarray(y_arr)
    if y_arr.shape[0] != mask.shape[0]:
        raise ValueError(
            f"optional tier length {y_arr.shape[0]} != mask length {mask.shape[0]}"
        )
    y_f = y_arr[mask]
    le = LabelEncoder()
    y_encoded_f = le.fit_transform(y_f)
    return y_f, y_encoded_f, le


def apply_cell_coords_column_rename(df, column_rename=None, auto_rename=True):
    """
    Rename CSV label columns to Hist2Pheno canonical names (in place on a copy).

    Args:
        df: cell coordinates + labels DataFrame.
        column_rename: dict ``{source_col: target_col}``. Renames only when ``source_col``
            exists and ``target_col`` is not already present (avoids clobbering NCRT CSVs).
        auto_rename: if True and ``column_rename`` is None, apply
            ``DEFAULT_CELL_COORD_COLUMN_RENAME_XENIUM`` when ``celltype`` is missing.

    Returns:
        (df_copy, rename_applied) where ``rename_applied`` maps source → target.
    """
    out = df.copy()
    rename_applied = {}

    if column_rename is None and auto_rename and REQUIRED_CELL_LABEL_COL not in out.columns:
        column_rename = dict(DEFAULT_CELL_COORD_COLUMN_RENAME_XENIUM)

    if not column_rename:
        return out, rename_applied

    for src, tgt in column_rename.items():
        if src not in out.columns:
            continue
        if tgt in out.columns and src != tgt:
            continue
        if src == tgt:
            continue
        out = out.rename(columns={src: tgt})
        rename_applied[src] = tgt

    return out, rename_applied


def resolve_label_column(name, column_rename=None):
    """Map a user-facing label column to its name after ``column_rename``."""
    if not column_rename or not name:
        return name
    return column_rename.get(name, name)


def load_cell_pixcoords(cell_coords_path, column_rename=None, auto_rename=True):
    """
    Load cell-type coordinate CSV for Hist2Pheno matching.

    Args:
        cell_coords_path: path to CSV.
        column_rename: optional ``{old_name: new_name}`` (see ``apply_cell_coords_column_rename``).
        auto_rename: apply Xenium defaults when ``celltype`` column is absent.
    """
    print("Loading celltype Ground Truth ...")
    if not os.path.exists(cell_coords_path):
        print(f"  File not found: {cell_coords_path}")
        return None

    celltype_pixel_NCRT_tumor1 = pd.read_csv(cell_coords_path)
    _c0 = celltype_pixel_NCRT_tumor1.columns[0]
    if str(_c0).startswith("Unnamed"):
        celltype_pixel_NCRT_tumor1 = celltype_pixel_NCRT_tumor1.drop(columns=[_c0])

    celltype_pixel_NCRT_tumor1, renamed = apply_cell_coords_column_rename(
        celltype_pixel_NCRT_tumor1,
        column_rename=column_rename,
        auto_rename=auto_rename,
    )
    if renamed:
        print(f"  Column rename applied: {renamed}")

    print(f"  Loaded {len(celltype_pixel_NCRT_tumor1)} cells")
    if REQUIRED_CELL_LABEL_COL not in celltype_pixel_NCRT_tumor1.columns:
        raise KeyError(
            f"Missing required column {REQUIRED_CELL_LABEL_COL!r}. "
            f"Available: {celltype_pixel_NCRT_tumor1.columns.tolist()}. "
            "Pass column_rename= e.g. {{'final_CT': 'celltype'}} or set auto_rename=True."
        )
    print(
        f"  Cell types: {celltype_pixel_NCRT_tumor1[REQUIRED_CELL_LABEL_COL].nunique()} unique types"
    )
    if renamed:
        celltype_pixel_NCRT_tumor1.attrs["_esccai_column_rename"] = renamed

    print("=" * 60 + "\n")
    return celltype_pixel_NCRT_tumor1


######################################################
# 2026.03.19 For model input: LLY Load HIPT embeddings
######################################################
def load_hist_embeddings(
    embedding_dir,
    celltype_df=None,
    coord_cols=None,
    tolerance=1.0,
    pth_prefix="sc_NCRT",
    force_full_scan=False,
    load_workers=0,
    selective_strategy="batch",
):
    """
    Load HIPT .pth embeddings.

    With ``celltype_df`` (recommended for huge folders): selective load, no full glob.
    With ``force_full_scan=True`` or no ``celltype_df``: ``glob('*.pth')`` + full load.

    If the folder has fewer than ``ESCCAI_EMBEDDING_PTH_CAP`` (default 400000) ``.pth``
    files, selective probing is **skipped** and those paths are loaded in one pass (avoids
    wasted time when filenames are not simple ``prefix_x_y.pth``). Full scan uses parallel
    ``torch.load`` when ``load_workers`` > 1 or auto (see ``load_embeddings``).

    Raise ``ESCCAI_EMBEDDING_PTH_CAP`` to force selective mode on very large folders.

    For flat fractional basenames (``sc_NCRT_10000.01…_6157.05….pth``), selective batch
    mode uses unique cell coordinates only after a short directory sample; set
    ``ESCCAI_SELECTIVE_LEGACY_PROBES=1`` to restore lattice probes. Sample size:
    ``ESCCAI_SELECTIVE_PREFLIGHT_MAX`` (default 512).
    """
    embedding_dir = Path(embedding_dir)
    pth_cap = int(os.environ.get("ESCCAI_EMBEDDING_PTH_CAP", "400000"))

    if celltype_df is not None and not force_full_scan:
        pth_files, hit_cap = _list_pth_paths_capped(embedding_dir, cap=pth_cap)
        if not hit_cap:
            print(
                f"Loading HIPT embeddings (full scan: {len(pth_files):,} .pth files, cap={pth_cap:,}; "
                "skipping selective path probe)."
            )
            embeddings_dict, failed_files = load_embeddings(
                pth_files, pth_prefix=pth_prefix, load_workers=load_workers
            )
        else:
            print(
                f"Loading HIPT embeddings from .pth files (selective mode; ≥{pth_cap:,} files)..."
            )
            embeddings_dict, failed_files = load_embeddings_selective(
                celltype_df,
                embedding_dir,
                coord_cols=coord_cols,
                tolerance=tolerance,
                pth_prefix=pth_prefix,
                load_workers=load_workers,
                selective_strategy=selective_strategy,
            )
            if len(embeddings_dict) == 0:
                print(
                    "  ⚠ Selective mode found 0 embeddings (on-disk names may not match "
                    "``{prefix}_{x}_{y}.pth`` probes). Falling back to full directory scan "
                    "(legacy behavior; can be slow for huge folders)."
                )
                pth_files = list(embedding_dir.glob("*.pth"))
                embeddings_dict, failed_files = load_embeddings(
                    pth_files, pth_prefix=pth_prefix, load_workers=load_workers
                )
    else:
        print("Loading HIPT embeddings from .pth files (full directory scan)...")
        pth_files = list(embedding_dir.glob("*.pth"))
        embeddings_dict, failed_files = load_embeddings(
            pth_files, pth_prefix=pth_prefix, load_workers=load_workers
        )

    if len(embeddings_dict) == 0:
        inferred = infer_pth_prefix_from_dir(embedding_dir)
        if inferred and inferred != pth_prefix:
            print(
                f"  ⚠ pth_prefix={pth_prefix!r} parsed 0 embeddings; "
                f"retrying with inferred prefix {inferred!r}"
            )
            pth_prefix = inferred
            pth_files = list(embedding_dir.glob("*.pth"))
            embeddings_dict, failed_files = load_embeddings(
                pth_files, pth_prefix=pth_prefix, load_workers=load_workers
            )

    print(f"  Successfully loaded {len(embeddings_dict)} embeddings")

    if len(failed_files) > 5:
        print(f"  ... and {len(failed_files) - 5} more files failed to load")
    if len(failed_files) > 0:
        print(f"  Total failed: {len(failed_files)} files")
    if len(embeddings_dict) > 0:
        sample_emb = list(embeddings_dict.values())[0]
        print(f"  Embedding dimension: {sample_emb.shape}")
        # print(f"  Sample embedding dtype: {sample_emb.dtype}")
    else:
        print("  ERROR: No embeddings loaded! Please check the .pth files.")

    print("="*60 + "\n")
    return embeddings_dict


######################################################
# 2026.06.24 LLY Matched features as AnnData (.h5ad) for Xenium lung
######################################################

XENIUM_H5AD_OBS_COLUMNS = (
    "CNiche",
    "TNiche",
    "final_CT",
    "final_lineage",
    "final_sublineage",
    "tma",
)


def adata_X_to_dense(X):
    """Return dense ``float`` ndarray from AnnData ``.X`` (sparse or dense)."""
    return _adata_X_to_dense(X)


def _adata_X_to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _unique_obs_names(names):
    names = pd.Index(names).astype(str)
    if not names.duplicated().any():
        return names
    seen = {}
    out = []
    for n in names:
        k = seen.get(n, 0)
        out.append(n if k == 0 else f"{n}_{k}")
        seen[n] = k + 1
    return pd.Index(out)


def build_matched_h5ad(
    cell_coords_path,
    matched_data,
    h5ad_path,
    *,
    obs_columns=XENIUM_H5AD_OBS_COLUMNS,
    cell_id_col="cell_id",
    spatial_cols=("x_centroid", "y_centroid"),
    spatial_he_cols=("X_pix_HE", "Y_pix_HE"),
):
    """
    Build ``{sample}_matched_features.h5ad`` from coordinate-matched rows.

    ``matched_data`` is the list of dicts returned by ``match_embeddings`` (must
    include ``idx`` and ``embedding``). Metadata is taken from the raw CSV at
    ``cell_coords_path`` (not the renamed training columns).
    """
    import anndata as ad

    if not matched_data:
        raise ValueError("matched_data is empty — cannot build h5ad.")

    raw_df = pd.read_csv(cell_coords_path)
    indices = [int(item["idx"]) for item in matched_data]
    if cell_id_col not in raw_df.columns:
        raise KeyError(f"{cell_id_col!r} not in {cell_coords_path}")

    obs_df = raw_df.iloc[indices].copy()
    obs_names = _unique_obs_names(obs_df[cell_id_col].astype(str))
    use_obs_cols = [c for c in obs_columns if c in obs_df.columns]
    missing_obs = [c for c in obs_columns if c not in obs_df.columns]
    if missing_obs:
        raise KeyError(
            f"CSV missing obs columns for h5ad: {missing_obs}. "
            f"Available: {list(raw_df.columns)}"
        )
    obs_df = obs_df[use_obs_cols]
    obs_df.index = obs_names

    for sc, label in (
        (spatial_cols, "spatial"),
        (spatial_he_cols, "spatial_HE"),
    ):
        if not all(c in raw_df.columns for c in sc):
            raise KeyError(f"CSV missing coordinate columns {sc} for obsm[{label!r}]")
    spatial = raw_df.iloc[indices][list(spatial_cols)].to_numpy(dtype=np.float64)
    spatial_he = raw_df.iloc[indices][list(spatial_he_cols)].to_numpy(dtype=np.float64)

    X = np.asarray([item["embedding"] for item in matched_data], dtype=np.float32)
    n_vars = X.shape[1]
    var_names = [str(i + 1) for i in range(n_vars)]

    adata = ad.AnnData(
        X=X,
        obs=obs_df,
        var=pd.DataFrame(index=var_names),
    )
    adata.obs_names = obs_names.astype(str)
    adata.var_names = var_names
    adata.obsm["spatial"] = spatial
    adata.obsm["spatial_HE"] = spatial_he
    adata.uns["cell_coords_path"] = str(cell_coords_path)
    adata.uns["matched_n_cells"] = int(adata.n_obs)
    adata.uns["embedding_dim"] = int(n_vars)

    h5ad_path = str(h5ad_path)
    os.makedirs(os.path.dirname(h5ad_path) or ".", exist_ok=True)
    adata.write_h5ad(h5ad_path)
    print(f"  ✓ Saved matched h5ad to {h5ad_path} ({adata.n_obs:,} cells × {n_vars} features)")
    return adata


def load_matched_h5ad(h5ad_path):
    """Load ``{sample}_matched_features.h5ad``."""
    import anndata as ad

    h5ad_path = str(h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    print(f"  ✓ Loaded h5ad: {h5ad_path} ({adata.n_obs:,} cells × {adata.n_vars} features)")
    return adata

########################################################
## 2026.06.24 LLY Matched features as AnnData (.h5ad) for Xenium lung (GT vs StarDist)
########################################################
H5AD_TO_NPZ_LABEL_KEYS = {
    "y": "final_CT",
    "y_level1": "final_lineage",
    "y_level12": "final_sublineage",
    "y_level3": "CNiche",
    "y_level4": "TNiche",
}


def load_matched_features_bundle(path):
    """
    Load matched features from ``.h5ad`` or legacy ``.npz`` as a plain dict.

    h5ad obs columns are mapped to the npz key names used by plotting helpers.
    """
    path = str(path)
    if path.endswith(".h5ad"):
        import anndata as ad

        adata = ad.read_h5ad(path)
        out = {"X": _adata_X_to_dense(adata.X)}
        for npz_key, obs_key in H5AD_TO_NPZ_LABEL_KEYS.items():
            if obs_key in adata.obs.columns:
                out[npz_key] = adata.obs[obs_key].to_numpy()
        if "spatial" in adata.obsm:
            out["X_coords"] = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        elif "spatial_HE" in adata.obsm:
            out["X_coords"] = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
        return out

    loaded = np.load(path, allow_pickle=True)
    return {k: loaded[k] for k in loaded.files}

########################################################
## 2026.06.24 LLY Encode labels with class names (from HEanno)
## find some problem with AUROC for celltype from StarDist
########################################################
def encode_labels_with_class_names(y_raw, class_names):
    """Map string (or numeric) labels to training ``class_names`` indices; unknown → -1."""
    name_to_idx = {str(c): i for i, c in enumerate(class_names)}
    y_arr = np.asarray(y_raw)
    if y_arr.dtype.kind in ("U", "S", "O"):
        return np.array([name_to_idx.get(str(v), -1) for v in y_arr], dtype=np.int64)
    y_num = y_arr.astype(np.int64, copy=False)
    out = np.full(y_num.shape, -1, dtype=np.int64)
    valid = (y_num >= 0) & (y_num < len(class_names))
    out[valid] = y_num[valid]
    return out
########################################################

def matched_features_npz_to_h5ad(
    npz_path,
    cell_coords_path,
    h5ad_path,
    *,
    tolerance=1.0,
    coord_cols=("X_pix_HE", "Y_pix_HE"),
    obs_columns=XENIUM_H5AD_OBS_COLUMNS,
    cell_id_col="cell_id",
):
    """
    Convert legacy ``{sample}_matched_features.npz`` to standard h5ad.

    Rows are linked to the CSV by nearest ``coord_cols`` (default HE pixel coords).
    """
    loaded = np.load(npz_path, allow_pickle=True)
    if "X_coords" not in loaded:
        raise KeyError(f"{npz_path} has no X_coords — cannot rebuild h5ad.")
    X = loaded["X"]
    X_coords = np.asarray(loaded["X_coords"], dtype=np.float64)

    raw_df = pd.read_csv(cell_coords_path)
    x_col, y_col = _resolve_cell_coord_columns(raw_df, coord_cols)
    csv_pts = raw_df[[x_col, y_col]].to_numpy(dtype=np.float64)
    valid = ~(np.isnan(csv_pts).any(axis=1))
    csv_pts = csv_pts[valid]
    csv_idx = np.nonzero(valid)[0]

    from scipy.spatial import cKDTree

    tree = cKDTree(csv_pts)
    dists, nn = tree.query(X_coords, k=1, distance_upper_bound=tolerance)
    matched_mask = nn < len(csv_idx)
    if not np.all(matched_mask):
        raise ValueError(
            f"Only {matched_mask.sum()} / {len(X_coords)} npz rows matched CSV within "
            f"tolerance={tolerance}. Delete npz and rebuild via match_hist2cell_h5ad."
        )

    matched_data = [
        {"idx": int(csv_idx[int(nn[i])]), "embedding": X[i]}
        for i in range(len(X))
    ]
    return build_matched_h5ad(
        cell_coords_path,
        matched_data,
        h5ad_path,
        obs_columns=obs_columns,
        cell_id_col=cell_id_col,
    )


def _matched_h5ad_usable(h5ad_path):
    if not os.path.exists(h5ad_path):
        return False
    try:
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path, backed="r")
        ok = all(c in adata.obs.columns for c in XENIUM_H5AD_OBS_COLUMNS)
        ok = ok and "spatial" in adata.obsm and "spatial_HE" in adata.obsm
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        return ok
    except Exception:
        return False


def match_hist2cell_h5ad(
    cell_coords_path,
    hist_embedding_dir,
    matched_h5ad_path,
    coord_cols=None,
    tolerance=1.0,
    pth_prefix="sc_NCRT",
    force_full_embedding_scan=False,
    embedding_load_workers=0,
    selective_embedding_strategy="batch",
    level1_name="celltype_level1",
    column_rename=None,
    auto_rename=True,
    force_rebuild=False,
    obs_columns=XENIUM_H5AD_OBS_COLUMNS,
    cell_id_col="cell_id",
    spatial_cols=("x_centroid", "y_centroid"),
    spatial_he_cols=("X_pix_HE", "Y_pix_HE"),
):
    """
    Match HIPT embeddings to cell CSV and save/load ``{sample}_matched_features.h5ad``.

    Replaces the npz path for Xenium lung: obs holds CSV labels; ``.X`` is UNI
    embeddings; ``obsm['spatial']`` = x/y centroid; ``obsm['spatial_HE']`` = HE pixels.

    For StarDist matching, pass ``coord_cols=('centroid_x', 'centroid_y')`` and
    ``spatial_cols=('centroid_x', 'centroid_y')`` so spatial plots use StarDist centroids.
    """
    matched_h5ad_path = str(matched_h5ad_path)
    if not force_rebuild and _matched_h5ad_usable(matched_h5ad_path):
        print(f"  ✓ matched h5ad cache found; loading {matched_h5ad_path}")
        return load_matched_h5ad(matched_h5ad_path)

    cell_coords_path = str(cell_coords_path)
    hist_embedding_dir = Path(hist_embedding_dir)

    celltype_df = load_cell_pixcoords(
        cell_coords_path,
        column_rename=column_rename,
        auto_rename=auto_rename,
    )
    if celltype_df is None:
        raise FileNotFoundError(cell_coords_path)

    embeddings_dict = load_hist_embeddings(
        hist_embedding_dir,
        celltype_df=celltype_df,
        coord_cols=coord_cols,
        tolerance=tolerance,
        pth_prefix=pth_prefix,
        force_full_scan=force_full_embedding_scan,
        load_workers=embedding_load_workers,
        selective_strategy=selective_embedding_strategy,
    )

    _rename_map = celltype_df.attrs.get("_esccai_column_rename", {}) or {}
    _level1_col = resolve_label_column(level1_name, _rename_map)
    if _level1_col not in celltype_df.columns and level1_name in celltype_df.columns:
        _level1_col = level1_name

    print("Matching embeddings with celltype data (this may take a few minutes)...")
    matched_data = match_embeddings(
        celltype_df,
        embeddings_dict,
        tolerance,
        coord_cols=coord_cols,
        level1_name=_level1_col,
    )
    print(f"  Matched {len(matched_data)} cells out of {len(celltype_df)}")
    print(f"  Match rate: {len(matched_data) / max(len(celltype_df), 1) * 100:.2f}%")
    if len(matched_data) == 0:
        raise ValueError(
            "Matched 0 cells — check pth_prefix / embedding directory / coordinates."
        )

    return build_matched_h5ad(
        cell_coords_path,
        matched_data,
        matched_h5ad_path,
        obs_columns=obs_columns,
        cell_id_col=cell_id_col,
        spatial_cols=spatial_cols,
        spatial_he_cols=spatial_he_cols,
    )


def prepare_data_from_matched_h5ad(
    adata,
    groups=None,
    min_l2_samples=2,
    *,
    y_col="final_CT",
    y_level1_col="final_lineage",
    y_level12_col="final_sublineage",
    y_level3_col="CNiche",
    y_level4_col="TNiche",
    spatial_he_key="spatial_HE",
):
    """
    Filter/encode labels from matched h5ad (same output dict as LOGO prep).

    Uses ``adata.obs`` hierarchy columns and ``adata.obsm[spatial_he_key]`` for coords.
    """
    X = _adata_X_to_dense(adata.X)
    for col in (y_col, y_level1_col, y_level12_col, y_level3_col, y_level4_col):
        if col not in adata.obs.columns:
            raise KeyError(f"adata.obs missing {col!r}")
    if spatial_he_key not in adata.obsm:
        raise KeyError(f"adata.obsm missing {spatial_he_key!r}")

    y = adata.obs[y_col].to_numpy()
    y_level1 = adata.obs[y_level1_col].to_numpy()
    y_level12 = adata.obs[y_level12_col].to_numpy()
    y_level3 = adata.obs[y_level3_col].to_numpy()
    y_level4 = adata.obs[y_level4_col].to_numpy()
    X_coords = np.asarray(adata.obsm[spatial_he_key], dtype=np.float64)

    if groups is None:
        groups = np.arange(adata.n_obs, dtype=np.int64)

    cv_data = prepare_data_leave_one_group_out(
        X,
        y,
        y_level1,
        groups=groups,
        min_l2_samples=min_l2_samples,
        y_level12=y_level12,
        y_level3=y_level3,
        y_level4=y_level4,
        X_coords=X_coords,
    )
    cv_data["adata"] = adata
    return cv_data


######################################################
# 2026.03.19 For model input: LLY Match HIPT embeddings with celltype pixel coordinates
######################################################
def match_hist2cell_matrix(
    cell_coords_path,
    hist_embedding_dir,
    matched_features_path,
    coord_cols=None,
    tolerance=1.0,
    pth_prefix="sc_NCRT",
    force_full_embedding_scan=False,
    embedding_load_workers=0,
    selective_embedding_strategy="batch",
    group_col=None,
    return_groups=False,
    level1_name="celltype_level1",
    column_rename=None,
    auto_rename=True,
    rebuild_if_missing_hierarchy=True,
    require_hierarchy_labels=False,
):
    """
    Match embeddings with celltype matrix and return aligned label arrays + coordinates.

    This wrapper supports both regular HE coordinates and StarDist coordinates
    with the same 3-path call signature:
      - cell_coords_path
      - hist_embedding_dir
      - matched_features_path

    Args:
        cell_coords_path: CSV path with at least ['celltype'] and coordinate columns.
        hist_embedding_dir: directory containing .pth embedding files.
        matched_features_path: output/input .npz cache path.
        coord_cols: optional tuple of coordinate columns. If None, auto-detect in
            prepare_matched_embeddings/match_embeddings:
            priority centroid_x/centroid_y, then X_pix_HE/Y_pix_HE.
        tolerance: coordinate matching tolerance in pixels.
        pth_prefix: e.g. ``sc_NCRT`` for ``sc_NCRT_x_y.pth``.
        force_full_embedding_scan: glob all ``*.pth`` (slow for huge dirs).
        embedding_load_workers: parallel loader threads in batch mode (0 = auto).
        selective_embedding_strategy: ``batch`` (default) or ``per_cell`` (slow).
        group_col: optional column for Leave-One-Group-Out (e.g. ``'TumorID'``); stored in npz.
        return_groups: if True, append ``groups`` as the last return value.
        level1_name: column used for ``y_level1`` (see ``prepare_matched_embeddings``).
        column_rename: optional ``{csv_col: canonical_col}`` before matching (see
            ``apply_cell_coords_column_rename``). When None and ``celltype`` is missing,
            Xenium defaults map ``final_CT`` → ``celltype``.
        auto_rename: apply Xenium default rename when ``celltype`` is absent.

    Returns:
        ``X, y, y_level1, y_level3, y_level4, y_level12, X_coords`` or with groups last.
        Extra level arrays are None if missing from cache (rebuild npz after CSV update).
    """
    # Normalize path-like inputs so callers can pass str or Path.
    cell_coords_path = str(cell_coords_path)
    hist_embedding_dir = Path(hist_embedding_dir)
    matched_features_path = str(matched_features_path)

    ## Load celltype data (ground truth Y)
    celltype_pixel_NCRT_tumor1 = load_cell_pixcoords(
        cell_coords_path,
        column_rename=column_rename,
        auto_rename=auto_rename,
    )
    if celltype_pixel_NCRT_tumor1 is None:
        raise FileNotFoundError(cell_coords_path)

    pm_kw = dict(
        tolerance=tolerance,
        coord_cols=coord_cols,
        group_col=group_col,
        return_groups=return_groups,
        level1_name=level1_name,
        rebuild_if_missing_hierarchy=rebuild_if_missing_hierarchy,
        require_hierarchy_labels=require_hierarchy_labels,
    )

    ## Load embeddings when there is no usable cache (missing npz or rebuild required).
    if _matched_npz_usable_as_cache(matched_features_path, rebuild_if_missing_hierarchy):
        print("  ✓ matched_features cache found; skipping .pth directory load.")
        embeddings_dict = {}
    else:
        if os.path.exists(matched_features_path):
            print("  ⚠ matched_features cache incomplete; loading .pth embeddings for rebuild...")
        embeddings_dict = load_hist_embeddings(
            hist_embedding_dir,
            celltype_df=celltype_pixel_NCRT_tumor1,
            coord_cols=coord_cols,
            tolerance=tolerance,
            pth_prefix=pth_prefix,
            force_full_scan=force_full_embedding_scan,
            load_workers=embedding_load_workers,
            selective_strategy=selective_embedding_strategy,
        )

    if return_groups:
        (
            X,
            y,
            y_level1,
            y_level3,
            y_level4,
            y_level12,
            X_coords_matched,
            groups,
        ) = prepare_matched_embeddings(
            celltype_pixel_NCRT_tumor1,
            embeddings_dict,
            matched_features_path,
            **pm_kw,
        )
    else:
        X, y, y_level1, y_level3, y_level4, y_level12, X_coords_matched = prepare_matched_embeddings(
            celltype_pixel_NCRT_tumor1,
            embeddings_dict,
            matched_features_path,
            **pm_kw,
        )
        groups = None
    if X_coords_matched is not None:
        print(f"  ✓ Coordinates loaded: {X_coords_matched.shape}")
    else:
        print(f"  ⚠ Coordinates not available (old format)")

    if return_groups:
        return X, y, y_level1, y_level3, y_level4, y_level12, X_coords_matched, groups
    return X, y, y_level1, y_level3, y_level4, y_level12, X_coords_matched




######################################################
# 2026.03.23 LLY Make PCF2HE2StarDist alignment, like make_PCF2HE_alignment
######################################################


def make_PCF2HE2StarDist_alignment(
    stardist_coords_path,
    qupath_corner_path,
    codex_meta_celltype_path,
    celltype_pixel_NCRT_path,
    therapy_data="NCRT",
    parent_value='tumor1',
    save_path=None,
    save_path4ViT=None,
    celltype_pixel_NCRT_df=None,
):
    """
    Align cell coordinates and types from StarDist, ROI, and GroundTruth annotation.
    Save matched cell coordinate files for downstream analysis.

    If celltype_pixel_NCRT_df is provided, it is used instead of reading
    celltype_pixel_NCRT_path (avoids a second full CSV read or writing a temp file).

    Returns:
        celltype_anno_df (pd.DataFrame): Celltype annotation dataframe.
        star_coords_in_roi (pd.DataFrame): StarDist coordinates within ROI.
        roi_cell_counts (pd.DataFrame): Cell counts per ROI.
    """

    # Load StarDist cell coordinates (no duplicates)
    star_coords = pd.read_csv(stardist_coords_path)

    # Load ROI coordinates from QuPath annotation
    roi_coords = pd.read_csv(qupath_corner_path)

    # Load celltype metadata and cell pixel coordinates
    celltype_anno_df = pd.read_csv(codex_meta_celltype_path)
    if celltype_pixel_NCRT_df is not None:
        celltype_pixel_NCRT = celltype_pixel_NCRT_df.copy()
    else:
        celltype_pixel_NCRT = pd.read_csv(celltype_pixel_NCRT_path)
    # print(celltype_pixel_NCRT.head())

    # Match StarDist cell coordinates to ROI
    star_coords_in_roi, roi_cell_counts = match_StarCoords2ROI(
        star_coords, roi_coords, therapy_data=therapy_data
    )

    ## Save csv files for downstream analysis
    ## Select GroundTruth celltype pixel coordinates
    if parent_value == 'all':
        celltype_pixel_NCRT_tumor1 = celltype_pixel_NCRT
        print("Cell number in GroundTruth (all TumorID):", celltype_pixel_NCRT_tumor1.shape[0])
    else:
        tumor_id = f"{therapy_data}_{parent_value}"
        celltype_pixel_NCRT_tumor1 = celltype_pixel_NCRT[celltype_pixel_NCRT['TumorID'] == tumor_id]
        print("Cell number in GroundTruth of tumor1:", celltype_pixel_NCRT_tumor1.shape[0])

    ## Select StarDist cell coordinates in tumor1 within ROI
    if parent_value == 'all':
        star_coords_in_roi1 = star_coords_in_roi
        print("Cell number in StarDist (all TumorID):", star_coords_in_roi1.shape[0])
    else:
        star_coords_in_roi1 = star_coords_in_roi[star_coords_in_roi['TumorID'] == tumor_id]
        print("Cell number in StarDist of tumor1:", star_coords_in_roi1.shape[0])

    if parent_value != "all":
        if celltype_pixel_NCRT_tumor1.shape[0] == 0:
            sample_ids = sorted(
                celltype_pixel_NCRT["TumorID"].dropna().astype(str).unique().tolist()
            )[:25]
            raise ValueError(
                f"No GroundTruth rows for TumorID={tumor_id!r}. "
                "Check parent_value and TumorID in the GroundTruth CSV (must match therapy_data, e.g. SA_tumor3). "
                f"Sample TumorID values: {sample_ids}"
            )
        if star_coords_in_roi1.shape[0] == 0:
            raise ValueError(
                f"No StarDist nuclei in ROI for TumorID={tumor_id!r}. "
                "Check StarDist CSV, QuPath ROI corners, and ROI Name vs parent_value."
            )

    ## Match GroundTruth celltype pixel coordinates to StarDist cell coordinates
    celltype_pixel_NCRT_stardist = match_celltype2stardist(
        celltype_pixel_NCRT_tumor1,
        star_coords_in_roi1,
        celltype_pixel_coords_cols=['X_pix_HE', 'Y_pix_HE'],
        stardist_pixel_coords_cols=['centroid_x', 'centroid_y']
    )

    ## Save matched celltype pixel coordinates
    saved = False
    if save_path is not None:
        celltype_pixel_NCRT_stardist.to_csv(save_path, index=False)
        print(f"Celltype pixel coordinates DataFrame saved to {save_path}")
        saved = True
    if save_path4ViT is not None:
        celltype_pixel_NCRT_stardist4ViT = celltype_pixel_NCRT_stardist.copy()
        celltype_pixel_NCRT_stardist4ViT.rename(
            columns={'centroid_x': 'pxl_row_in_fullres', 'centroid_y': 'pxl_col_in_fullres'},
            inplace=True
        )
        celltype_pixel_NCRT_stardist4ViT.to_csv(save_path4ViT, index=False)
        print(f"Celltype pixel coordinates DataFrame for ViT saved to {save_path4ViT}")
        saved = True
    if not saved:
        print("Celltype pixel coordinates DataFrame not saved.")

    return celltype_anno_df, roi_cell_counts, star_coords_in_roi, celltype_pixel_NCRT




    
######################################
# 2026.03.23 LLY Save level2 + level1 validation metrics for HCE-lambda comparison.
######################################
import os
import pandas as pd
from datetime import datetime

def collect_hce_tier_metrics_extras(logo_summary=None, insample=None, num_classes=None):
    """
    Build flat CSV columns for every label tier (CV fold means + in-sample metrics).

    Parameters
    ----------
    logo_summary : dict, optional
        Output of ``train_hce_stratified_kfold_cv`` / LOGO (keys like ``l12_macro_f1_mean``).
    insample : dict, optional
        ``{tier: {"accuracy", "macro_f1", "weighted_f1"}}`` with tiers
        ``l2``, ``l1``, ``l1_head``, ``l12``, ``l3``, ``l4``.
    num_classes : dict, optional
        ``{tier: int}`` class counts per tier.
    """
    row = {}
    tiers = ("l2", "l1", "l1_head", "l12", "l3", "l4")
    metrics = ("accuracy", "macro_f1", "weighted_f1")

    if logo_summary is not None:
        for tier in tiers:
            for met in metrics:
                key = f"{tier}_{met}"
                mean_k = f"{key}_mean"
                std_k = f"{key}_std"
                if mean_k in logo_summary:
                    row[f"cv_{key}"] = logo_summary[mean_k]
                if std_k in logo_summary:
                    row[f"cv_{key}_std"] = logo_summary[std_k]

    if insample is not None:
        for tier, triplet in insample.items():
            if not isinstance(triplet, dict):
                continue
            for met in metrics:
                if met in triplet:
                    row[f"insample_{tier}_{met}"] = triplet[met]

    if num_classes is not None:
        for tier, n in num_classes.items():
            if n is not None:
                row[f"num_{tier}_classes"] = int(n)

    return row


def save_hce_validation_metrics(
    val_acc,
    val_macro_f1,
    val_weighted_f1,
    val_level1_acc,
    val_level1_macro_f1,
    val_level1_weighted_f1,
    class_names=None,
    class_names_level1=None,
    best_epoch=None,
    hce_lambda=None,
    hce_w1=None,
    hce_w2=None,
    hce_w12=None,
    hce_w_l12head=None,
    hce_w_l3=None,
    hce_w_l4=None,
    therapy_data=None,
    metrics_csv_path=None,
    extra_metrics=None,
):
    """Append one HCE validation row to CSV (legacy L2/L1 columns + optional ``extra_metrics``)."""
    metrics_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "therapy_data": therapy_data,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "hce_lambda": float(hce_lambda) if hce_lambda is not None else None,
        "hce_w1": float(hce_w1) if hce_w1 is not None else None,
        "hce_w2": float(hce_w2) if hce_w2 is not None else None,
        "hce_w12": float(hce_w12) if hce_w12 is not None else None,
        "hce_w_l12head": float(hce_w_l12head) if hce_w_l12head is not None else None,
        "hce_w_l3": float(hce_w_l3) if hce_w_l3 is not None else None,
        "hce_w_l4": float(hce_w_l4) if hce_w_l4 is not None else None,
        "val_l2_accuracy": float(val_acc),
        "val_l2_macro_f1": float(val_macro_f1),
        "val_l2_weighted_f1": float(val_weighted_f1),
        "val_l1_accuracy": float(val_level1_acc),
        "val_l1_macro_f1": float(val_level1_macro_f1),
        "val_l1_weighted_f1": float(val_level1_weighted_f1),
        "num_l2_classes": int(len(class_names)) if class_names is not None else None,
        "num_l1_classes": int(len(class_names_level1)) if class_names_level1 is not None else None,
    }
    if extra_metrics:
        metrics_row.update(extra_metrics)
    metrics_df = pd.DataFrame([metrics_row])

    if metrics_csv_path is None:
        raise ValueError("metrics_csv_path must be provided.")

    if os.path.exists(metrics_csv_path):
        existing_df = pd.read_csv(metrics_csv_path)
        combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    else:
        combined_df = metrics_df.copy()

    combined_df.to_csv(metrics_csv_path, index=False)

    print("Current run metrics:")
    print(metrics_df)
    print(f"\nSaved comparison table to: {metrics_csv_path}")
    print(f"Total rows: {len(combined_df)}")
    return metrics_df, combined_df


######################################
# 2026.03.23 LLY Split data into train/test sets with label encoding and feature scaling.
######################################
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def split_train_test(
    X,
    y,
    y_level1,
    test_size=0.3,
    random_state=42
):
    """
    Split data into train/test sets with label encoding and feature scaling.
    Handles level2 (fine-grained) and level1 (coarse-grained) labels.
    Filters out level2 classes with fewer than 2 samples to avoid stratified split errors.
    Re-encodes filtered labels to contiguous ids.
    Standardizes features.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray or list): Level2 labels (fine-grained).
        y_level1 (np.ndarray or list): Level1 labels (coarse-grained).
        test_size (float): Test set proportion.
        random_state (int): Random seed.

    Returns:
        dict: Contains all splits, encoders, class names, and scaler.
    """

    # Encode raw level2 labels
    le_raw_l2 = LabelEncoder()
    y_encoded_raw = le_raw_l2.fit_transform(y)

    # Encode level1 labels
    le_level1 = LabelEncoder()
    y_level1_encoded = le_level1.fit_transform(y_level1)
    class_names_level1 = le_level1.classes_

    # Filter out level2 classes with <2 samples
    unique_labels, counts = np.unique(y_encoded_raw, return_counts=True)
    keep_labels = unique_labels[counts >= 2]
    mask = np.isin(y_encoded_raw, keep_labels)

    X_f = X[mask]
    y_f = np.array(y)[mask]
    y_level1_f = np.array(y_level1)[mask]
    y_level1_encoded_f = y_level1_encoded[mask]

    # Re-encode filtered level2 labels to contiguous ids [0, ..., K-1]
    le = LabelEncoder()
    y_encoded_f = le.fit_transform(y_f)
    class_names = le.classes_

    # If only 1 class remains, stratify must be disabled
    strat = y_encoded_f if len(np.unique(y_encoded_f)) > 1 else None

    # Split data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_encoded,
        y_test_encoded,
        y_train_level1,
        y_test_level1,
        y_train_level1_encoded,
        y_test_level1_encoded,
    ) = train_test_split(
        X_f,
        y_f,
        y_encoded_f,
        y_level1_f,
        y_level1_encoded_f,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Print summary
    print(f"\nTrain/Test split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Feature dimension: {X_train.shape[1]}")
    print(f"  Level2 classes: {len(class_names)}")
    print(f"  Level1 classes: {len(class_names_level1)}")
    print("  Features standardized")

    # Return all useful outputs in a dictionary
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_encoded': y_train_encoded,
        'y_test_encoded': y_test_encoded,
        'y_train_level1': y_train_level1,
        'y_test_level1': y_test_level1,
        'y_train_level1_encoded': y_train_level1_encoded,
        'y_test_level1_encoded': y_test_level1_encoded,
        'class_names': class_names,
        'class_names_level1': class_names_level1,
        'le_level2': le,
        'le_level1': le_level1,
        'scaler': scaler,
        'mask': mask,
        'y_encoded_f': y_encoded_f,
        'y_level1_encoded_f': y_level1_encoded_f,
    }


def spatial_tile_groups_for_logo(x_he, y_he, nx=12, ny=12):
    """
    Build group ids by tiling HE pixel space into ``nx`` by ``ny`` equal-width bins on each axis.

    Use when a metadata column such as ``TumorID`` is constant for the whole table (e.g. all rows
    ``NCRT_tumor1``) so true leave-one-patient/slide CV is impossible without another id column.
    This gives spatially disjoint groups — a practical proxy for local generalization on one WSI.
    """
    x_he = np.asarray(x_he, dtype=np.float64).ravel()
    y_he = np.asarray(y_he, dtype=np.float64).ravel()
    if x_he.shape[0] != y_he.shape[0]:
        raise ValueError("x_he and y_he must have the same length")

    def _axis_bin_ids(v, n_bins):
        n_bins = max(2, int(n_bins))
        vmin, vmax = float(np.min(v)), float(np.max(v))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros(len(v), dtype=np.int64)
        edges = np.linspace(vmin, vmax, n_bins + 1)
        edges[-1] = np.nextafter(edges[-1], np.inf)
        ids = np.digitize(v, edges[1:-1], right=False)
        return np.clip(ids, 0, n_bins - 1).astype(np.int64)

    ix = _axis_bin_ids(x_he, nx)
    iy = _axis_bin_ids(y_he, ny)
    return np.array([f"tile_x{int(a)}_y{int(b)}" for a, b in zip(ix, iy)], dtype=object)


def prepare_data_leave_one_group_out(
    X,
    y,
    y_level1,
    groups,
    min_l2_samples=2,
    y_level12=None,
    y_level3=None,
    y_level4=None,
    X_coords=None,
):
    """
    Same rare-class filtering and L2/L1 encoding as ``split_train_test``, but keep the full
    filtered matrix for group-wise CV (no train/test split, no scaling — scaling is per fold).

    ``groups`` must align row-wise with ``X`` (e.g. ``TumorID`` per cell). Sklearn's
    ``LeaveOneGroupOut`` then holds out one group at a time, analogous to
    ``cross_val_predict(..., cv=LeaveOneGroupOut(), groups=groups)`` for sklearn estimators.

    Optional ``y_level12`` / ``y_level3`` / ``y_level4`` (unicode label arrays from
    ``prepare_matched_embeddings``) are encoded on the same rare-L2 mask when provided.

    Returns
    -------
    dict with keys: X_f, y_f, y_level1_f, y_encoded_f, y_level1_encoded_f, groups_f,
    class_names, class_names_level1, le_level2, le_level1, mask,
    and optional tier keys (``y_level12_f``, ``y_level12_encoded_f``, ``class_names_level12``, …).
    """
    groups = np.asarray(groups)
    if groups.shape[0] != X.shape[0]:
        raise ValueError(
            f"groups length {groups.shape[0]} != X.shape[0] {X.shape[0]}"
        )

    le_raw_l2 = LabelEncoder()
    y_encoded_raw = le_raw_l2.fit_transform(y)

    le_level1 = LabelEncoder()
    y_level1_encoded = le_level1.fit_transform(y_level1)
    class_names_level1 = le_level1.classes_

    unique_labels, counts = np.unique(y_encoded_raw, return_counts=True)
    keep_labels = unique_labels[counts >= min_l2_samples]
    mask = np.isin(y_encoded_raw, keep_labels)

    X_f = X[mask]
    y_f = np.array(y)[mask]
    y_level1_f = np.array(y_level1)[mask]
    y_level1_encoded_f = y_level1_encoded[mask]
    groups_f = groups[mask]

    le = LabelEncoder()
    y_encoded_f = le.fit_transform(y_f)
    class_names = le.classes_

    out = {
        "X_f": X_f,
        "y_f": y_f,
        "y_level1_f": y_level1_f,
        "y_encoded_f": y_encoded_f,
        "y_level1_encoded_f": y_level1_encoded_f,
        "groups_f": groups_f,
        "class_names": class_names,
        "class_names_level1": class_names_level1,
        "le_level2": le,
        "le_level1": le_level1,
        "mask": mask,
    }

    if X_coords is not None:
        X_coords = np.asarray(X_coords, dtype=np.float64)
        if X_coords.shape[0] != mask.shape[0]:
            raise ValueError(
                f"X_coords length {X_coords.shape[0]} != X length {mask.shape[0]}"
            )
        out["X_coords_f"] = X_coords[mask]

    tier_specs = (
        ("y_level12", y_level12, "level12"),
        ("y_level3", y_level3, "level3"),
        ("y_level4", y_level4, "level4"),
    )
    for src_key, src_arr, short in tier_specs:
        if src_arr is None:
            continue
        y_t_f, y_t_enc, le_t = _encode_optional_tier_after_mask(src_arr, mask)
        out[f"y_{short}_f"] = y_t_f
        out[f"y_{short}_encoded_f"] = y_t_enc
        out[f"class_names_{short}"] = le_t.classes_
        out[f"le_{short}"] = le_t

    n_grp = len(np.unique(groups_f))
    print(f"\nLeave-one-group-out prep (filtered data):")
    print(f"  Samples: {X_f.shape[0]}, features: {X_f.shape[1]}")
    print(f"  Level2 classes: {len(class_names)}")
    print(f"  Level1 classes: {len(class_names_level1)}")
    if "y_level12_encoded_f" in out:
        print(f"  Level12 classes: {len(out['class_names_level12'])}")
    if "y_level3_encoded_f" in out:
        print(f"  Level3 (CNiche) classes: {len(out['class_names_level3'])}")
    if "y_level4_encoded_f" in out:
        print(f"  Level4 (TNiche) classes: {len(out['class_names_level4'])}")
    if "X_coords_f" in out:
        print(f"  Coordinates: {out['X_coords_f'].shape}")
    print(f"  Groups (for LOGO): {n_grp} unique")

    return out


######################################
# 2026.03.23 LLY Create hierarchical DataLoaders from split_train_test(...) output dict.
######################################
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

def result_has_extra_head_labels(result):
    """True when train/val splits include level12 + CNiche + TNiche encoded targets."""
    return (
        "y_train_level12_encoded" in result
        and "y_train_level3_encoded" in result
        and "y_train_level4_encoded" in result
    )


def loader_train_test(
    result,
    batch_size_cuda=1024,
    batch_size_cpu=256,
    num_workers_cuda=4,
    num_workers_cpu=0,
    pin_memory_cuda=True,
    pin_memory_cpu=False,
    seed=42,
    train_balance_sampler=False,
    train_balance_sampler_mode="global_l2",
):
    """
    Create hierarchical DataLoaders from split_train_test(...) output dict.

    When ``result`` contains ``y_train_level12_encoded``, ``y_train_level3_encoded``, and
    ``y_train_level4_encoded``, loaders yield six targets for five-head training; otherwise
    the legacy three-tuple ``(x, y_l2, y_l1)`` is used (NCRT / esccAI).

    train_balance_sampler : bool
        If True, train batches are drawn with ``WeightedRandomSampler``.
    train_balance_sampler_mode : {"global_l2", "hierarchical_l1_l2"}
        ``global_l2`` uses inverse frequency on level2 labels across the full
        training set (legacy behavior). ``hierarchical_l1_l2`` first balances
        level1 parents, then balances level2 children within each parent:
        sample weight = ``1 / count(level1) * 1 / count(level2 within level1)``.
        This reduces over-sampling drift from very large level1 groups.
    """
    X_train_scaled = result["X_train_scaled"]
    X_test_scaled = result["X_test_scaled"]
    y_train_encoded = result["y_train_encoded"]
    y_test_encoded = result["y_test_encoded"]
    y_train_level1_encoded = result["y_train_level1_encoded"]
    y_test_level1_encoded = result["y_test_level1_encoded"]

    use_cuda = torch.cuda.is_available()
    batch_size = batch_size_cuda if use_cuda else batch_size_cpu
    num_workers = num_workers_cuda if use_cuda else num_workers_cpu
    pin_memory = pin_memory_cuda if use_cuda else pin_memory_cpu

    g = torch.Generator().manual_seed(seed)

    X_train_tensor = torch.as_tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.as_tensor(X_test_scaled, dtype=torch.float32)
    y_train_l2_tensor = torch.as_tensor(y_train_encoded, dtype=torch.long)
    y_test_l2_tensor = torch.as_tensor(y_test_encoded, dtype=torch.long)
    y_train_l1_tensor = torch.as_tensor(y_train_level1_encoded, dtype=torch.long)
    y_test_l1_tensor = torch.as_tensor(y_test_level1_encoded, dtype=torch.long)

    use_extra = result_has_extra_head_labels(result)
    use_spatial = bool(result.get("use_spatial_context"))
    if use_spatial:
        if "train_global_indices" not in result or "val_global_indices" not in result:
            raise ValueError(
                "use_spatial_context=True requires train_global_indices and val_global_indices in result"
            )
        train_gidx = torch.as_tensor(result["train_global_indices"], dtype=torch.long)
        val_gidx = torch.as_tensor(result["val_global_indices"], dtype=torch.long)

    if use_extra:
        y_train_l12_tensor = torch.as_tensor(result["y_train_level12_encoded"], dtype=torch.long)
        y_test_l12_tensor = torch.as_tensor(result["y_test_level12_encoded"], dtype=torch.long)
        y_train_l3_tensor = torch.as_tensor(result["y_train_level3_encoded"], dtype=torch.long)
        y_test_l3_tensor = torch.as_tensor(result["y_test_level3_encoded"], dtype=torch.long)
        y_train_l4_tensor = torch.as_tensor(result["y_train_level4_encoded"], dtype=torch.long)
        y_test_l4_tensor = torch.as_tensor(result["y_test_level4_encoded"], dtype=torch.long)
        if use_spatial:
            train_dataset = TensorDataset(
                X_train_tensor,
                y_train_l2_tensor,
                y_train_l1_tensor,
                y_train_l12_tensor,
                y_train_l3_tensor,
                y_train_l4_tensor,
                train_gidx,
            )
            val_dataset = TensorDataset(
                X_test_tensor,
                y_test_l2_tensor,
                y_test_l1_tensor,
                y_test_l12_tensor,
                y_test_l3_tensor,
                y_test_l4_tensor,
                val_gidx,
            )
        else:
            train_dataset = TensorDataset(
                X_train_tensor,
                y_train_l2_tensor,
                y_train_l1_tensor,
                y_train_l12_tensor,
                y_train_l3_tensor,
                y_train_l4_tensor,
            )
            val_dataset = TensorDataset(
                X_test_tensor,
                y_test_l2_tensor,
                y_test_l1_tensor,
                y_test_l12_tensor,
                y_test_l3_tensor,
                y_test_l4_tensor,
            )
    elif use_spatial:
        train_dataset = TensorDataset(
            X_train_tensor, y_train_l2_tensor, y_train_l1_tensor, train_gidx
        )
        val_dataset = TensorDataset(
            X_test_tensor, y_test_l2_tensor, y_test_l1_tensor, val_gidx
        )
    else:
        train_dataset = TensorDataset(X_train_tensor, y_train_l2_tensor, y_train_l1_tensor)
        val_dataset = TensorDataset(X_test_tensor, y_test_l2_tensor, y_test_l1_tensor)

    if train_balance_sampler:
        yc = np.asarray(y_train_encoded, dtype=np.int64)
        yl1 = np.asarray(y_train_level1_encoded, dtype=np.int64)
        if train_balance_sampler_mode == "global_l2":
            max_c = int(yc.max()) + 1
            class_counts = np.bincount(yc, minlength=max_c).astype(np.float64)
            class_counts = np.maximum(class_counts, 1.0)
            w_per_class = 1.0 / class_counts
            sample_weights = w_per_class[yc]
        elif train_balance_sampler_mode == "hierarchical_l1_l2":
            max_l1 = int(yl1.max()) + 1
            l1_counts = np.bincount(yl1, minlength=max_l1).astype(np.float64)
            l1_counts = np.maximum(l1_counts, 1.0)
            sample_weights = np.empty(len(yc), dtype=np.float64)
            for parent in np.unique(yl1):
                parent_mask = yl1 == parent
                yc_parent = yc[parent_mask]
                max_c_parent = int(yc_parent.max()) + 1
                l2_counts_parent = np.bincount(
                    yc_parent, minlength=max_c_parent
                ).astype(np.float64)
                l2_counts_parent = np.maximum(l2_counts_parent, 1.0)
                sample_weights[parent_mask] = (
                    (1.0 / l1_counts[int(parent)])
                    * (1.0 / l2_counts_parent[yc_parent])
                )
        else:
            raise ValueError(
                "train_balance_sampler_mode must be one of: global_l2, hierarchical_l1_l2"
            )
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=g,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=g,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )

    val_eval_dataset = (
        TensorDataset(X_test_tensor, y_test_l2_tensor, val_gidx)
        if use_spatial
        else TensorDataset(X_test_tensor, y_test_l2_tensor)
    )
    val_loader_eval = DataLoader(
        val_eval_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    if train_balance_sampler:
        if train_balance_sampler_mode == "global_l2":
            print(
                "  train_balance_sampler=True: WeightedRandomSampler "
                "(inverse L2 class frequency)"
            )
        else:
            print(
                "  train_balance_sampler=True: WeightedRandomSampler "
                "(hierarchical L1 -> L2 balancing)"
            )
        unique_l1_values = np.unique(yl1)
        global_w_min = float(np.min(sample_weights))
        global_w_max = float(np.max(sample_weights))
        print(
            "  Sampler weight summary: "
            f"global_range=[{global_w_min:.6e}, {global_w_max:.6e}]"
        )
        for parent in unique_l1_values:
            parent_mask = yl1 == parent
            parent_weights = sample_weights[parent_mask]
            parent_l2 = yc[parent_mask]
            l2_ranges = []
            for child in np.unique(parent_l2):
                child_weights = parent_weights[parent_l2 == child]
                l2_ranges.append(
                    f"L2={int(child)}:[{float(np.min(child_weights)):.3e},{float(np.max(child_weights)):.3e}]"
                )
            print(
                "    "
                f"L1={int(parent)} weight_range=[{float(np.min(parent_weights)):.3e},"
                f"{float(np.max(parent_weights)):.3e}] "
                + " ".join(l2_ranges)
            )
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Feature dimension: {X_train_scaled.shape[1]}")

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "val_eval_dataset": val_eval_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "val_loader_eval": val_loader_eval,
        "batch_size": batch_size,
        "feature_dim": X_train_scaled.shape[1],
        "use_cuda": use_cuda,
        # pass-through for downstream training/validation calls
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_encoded": y_train_encoded,
        "y_test_encoded": y_test_encoded,
        "y_train_level1_encoded": y_train_level1_encoded,
        "y_test_level1_encoded": y_test_level1_encoded,
        "y_encoded_f": result["y_encoded_f"],
        "y_level1_encoded_f": result["y_level1_encoded_f"],
        "y_train": result["y_train"],
        "y_test": result["y_test"],
        "y_train_level1": result["y_train_level1"],
        "y_test_level1": result["y_test_level1"],
    }




######################################
# 2026.04.14 LLY Load the first .pth file in the specified directory and extract the tensor.
######################################
from pathlib import Path
import torch
def first_pth_tensor(dir_path):
    """
    Load the first .pth file in the specified directory and extract the tensor.
    Supports .pth files containing a tensor, tuple, or dict.
    Returns: tensor object
    """
    hist_embedding_dir = Path(dir_path)
    pth_files = list(hist_embedding_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError("No .pth files found in the directory.")
    pth_file = pth_files[0]
    print(f"Found file: {pth_file}")

    image_tensor = torch.load(pth_file)
    print(f"Loaded type: {type(image_tensor)}")

    # Type checking and extraction
    if isinstance(image_tensor, torch.Tensor):
        tensor = image_tensor
    elif isinstance(image_tensor, tuple):
        tensor = image_tensor[0]
    elif isinstance(image_tensor, dict):
        # You can modify the key selection according to your needs
        key = list(image_tensor.keys())[0]
        print(f"First key in dict: {key}")
        tensor = image_tensor[key]
    else:
        raise TypeError("Unknown .pth file structure")

    print(f"tensor.shape of the first .pth file: {tensor.shape}")
    return tensor