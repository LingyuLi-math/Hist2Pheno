## 2025.05.20 LLY adjust the code, part of it is from SpatialScopeNS.py
## 2026.02.03 LLY add the annotation and clean code, rename  SpatialScopeNS to FineST_StarDist_nuclei


import scanpy as sc
import squidpy as sq
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import argparse
import anndata
import pandas as pd
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from FineST.utils import *  # FineST package utils
from matplotlib.path import Path
import numpy as np
from skimage import draw, measure, io


class FineST_StarDist_nuclei:
    """
    FineST StarDist Nuclei Segmentation using predict_instances_big for large images.
    Supports ROI or full-image mode, integrates ST data if provided.
    """

    def __init__(self, tissue, out_dir, 
                 roi_path, img_path, adata_path=None, 
                 prob_thresh=0.5, max_cell_number=20, min_counts=500, tilesize=None):
        self.tissue = tissue
        self.out_dir = out_dir 
        self.roi_path = roi_path
        self.img_path = img_path
        self.adata_path = adata_path
        self.prob_thresh = prob_thresh
        self.max_cell_number = max_cell_number
        self.min_counts = min_counts
        self.tilesize = tilesize

        os.makedirs(out_dir, exist_ok=True)
        tissue_dir = os.path.join(out_dir, tissue)
        os.makedirs(tissue_dir, exist_ok=True)
        self.out_dir = tissue_dir

        self.loggings = configure_logging(os.path.join(self.out_dir, 'logs'))

        if self.roi_path is not None:
            if self.adata_path is None:
                raise ValueError("roi_path requires adata_path to be provided")
            self.ST_Data, self.Img_Data = self.crop_img_adata()
            self.LoadData(self.ST_Data, self.Img_Data, self.min_counts, roi_path=self.roi_path)
        else:
            self.ST_Data = self.adata_path
            self.Img_Data = self.img_path
            self.LoadData(self.ST_Data, self.Img_Data, self.min_counts, roi_path=None)


    def create_mask(self, polygon, shape):
        polygon = polygon.iloc[:, -2:].values
        polygon = np.clip(polygon, a_min=0, a_max=None)
        rr, cc = draw.polygon(polygon[:, 0], polygon[:, 1], shape) 
        mask = np.zeros(shape, dtype=bool)
        mask[rr, cc] = True
        return mask, polygon


    def crop_img_adata(self):
        roi_coords = pd.read_csv(self.roi_path)
        img = io.imread(self.img_path)
        mask, roi_coords = self.create_mask(roi_coords, img.shape[:2])
        props = measure.regionprops_table(mask.astype(int), properties=('bbox',))

        minr = props['bbox-0'][0]
        minc = props['bbox-1'][0]
        maxr = props['bbox-2'][0]
        maxc = props['bbox-3'][0]

        cropped_img = img[minr:maxr, minc:maxc]
        io.imsave(os.path.join(self.out_dir, 'cropped_img.tif'), cropped_img)

        adata = sc.read_h5ad(self.adata_path)
        roi_yx = roi_coords[:, [1, 0]]   
        adata_roi = adata[Path(roi_yx).contains_points(adata.obsm["spatial"]), :].copy()

        if len(roi_coords) > 2 and roi_coords[2][0] == 0: 
            adata_roi.obsm["spatial"] = adata_roi.obsm["spatial"] - np.array([roi_coords[0][1], 0])
        else: 
            adata_roi.obsm["spatial"] = adata_roi.obsm["spatial"] - np.array([roi_coords[0][1], roi_coords[0][0]])

        adata_roi.write(os.path.join(self.out_dir, 'adata_roi.h5ad'))
        return adata_roi, cropped_img


    def LoadData(self, ST_Data, Img_Data, min_counts, roi_path=None):
        if ST_Data is not None:
        if roi_path is not None:
            sp_adata = ST_Data
        else:
            sp_adata = anndata.read_h5ad(ST_Data)
        sp_adata.obs_names_make_unique()
        sp_adata.var_names_make_unique()
        self.sp_adata = sp_adata
        else:
            self.sp_adata = None
        
        if roi_path is not None:
            img = sq.im.ImageContainer(Img_Data)
        else:
            image = plt.imread(Img_Data)
            img = sq.im.ImageContainer(image)

        crop = img.crop_corner(0, 0)
        self.image = crop


    @staticmethod
    def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=None, tilesize=None):
        axis_norm = (0,1,2) if img.ndim == 3 else (0,1)
        img = normalize(img, 1, 99.8, axis=axis_norm)
        model = StarDist2D.from_pretrained('2D_versatile_he')

        # Determine axes based on image dimensions
        # For 3D images (H, W, C), axes should be 'YXC'
        # For 2D images (H, W), axes should be 'YX'
        if img.ndim == 3:
            axes = 'YXC'  # Height, Width, Channels
            n_spatial_dims = 2  # Only Y and X are spatial dimensions
        else:
            axes = 'YX'   # Height, Width
            n_spatial_dims = 2

        # Use predict_instances_big for large images with tiling
        if tilesize is not None and tilesize > 0:
            # According to stardist/models/base.py:
            # - If block_size/min_overlap/context are scalars, they are automatically expanded to all dimensions
            # - For channel axis 'C', the code automatically sets block_size[i] = img.shape[i] and min_overlap[i] = context[i] = 0
            # - If context=None, the function automatically uses recommended values from _axes_tile_overlap()
            # So we can pass scalar values and let the function handle it automatically
            block_size = tilesize  # Scalar value, will be expanded automatically
            min_overlap = int(tilesize * 0.1)  # 10% overlap, scalar value
            context = None  # Use automatic recommended context value (typically ~94 for StarDist2D)
            
            print(f"Using predict_instances_big: image size {img.shape}, axes={axes}, block_size={block_size}, min_overlap={min_overlap}, context={context} (auto)")
            
            labels, _ = model.predict_instances_big(
                img,
                axes=axes,
                block_size=block_size,
                min_overlap=min_overlap,
                context=context,
                nms_thresh=nms_thresh,
                prob_thresh=prob_thresh
            )
        else:
            # For small images, use regular predict_instances
            labels, _ = model.predict_instances(
                img,
                nms_thresh=nms_thresh,
                prob_thresh=prob_thresh
            )
        return labels

    
    @staticmethod
    def DissectSegRes(df):
        tmps = []
        for row in df.iterrows():
            if row[1]['segmentation_label'] == 0:
                continue
            for idx,i in enumerate(row[1]['segmentation_centroid']):
                tmps.append(list(i)+[row[0],row[0]+'_{}'.format(idx),row[1]['segmentation_label']])
        return pd.DataFrame(tmps,columns=['x','y','spot_index','cell_index','cell_nums'])  

        
    def NucleiSegmentation(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        _ = StarDist2D.from_pretrained('2D_versatile_he')

        def stardist_with_tilesize(img, nms_thresh=None, prob_thresh=None):
            return self.stardist_2D_versatile_he(img, nms_thresh=nms_thresh, prob_thresh=prob_thresh, tilesize=self.tilesize)

        sq.im.segment(
            img=self.image,
            layer="image",
            channel=None,
            method=stardist_with_tilesize,
            layer_added='segmented_stardist_default',
            prob_thresh=self.prob_thresh
        )
        
        if self.sp_adata is not None:
        features_kwargs = {
            "segmentation": {
                "label_layer": "segmented_stardist_default",
                "props": ["label", "centroid"],
                "channels": [1, 2],
            }
        }
        
        sq.im.calculate_image_features(
            self.sp_adata,
            self.image,
            layer="image",
            key_added="image_features",
            features_kwargs=features_kwargs,
            features="segmentation",
            mask_circle=True,
        )
        
        df_cells = self.sp_adata.obsm['image_features'].copy().astype(object)
        for row in df_cells.iterrows():
            if row[1]['segmentation_label']>self.max_cell_number:
                    df_cells.at[row[0],'segmentation_label'] = self.max_cell_number
                df_cells.at[row[0],'segmentation_centroid'] = row[1]['segmentation_centroid'][:self.max_cell_number]

        self.sp_adata.obsm['image_features'] = df_cells       
        self.sp_adata.obs["cell_count"] = self.sp_adata.obsm["image_features"]["segmentation_label"].astype(int)
        
            # visualization
        fig, axes = plt.subplots(1,3,figsize=(20,6),dpi=300)
        self.image.show("image", ax=axes[0])
        _ = axes[0].set_title("H&E")
        self.image.show("segmented_stardist_default", cmap="jet", interpolation="none", ax=axes[1])
        _ = axes[1].set_title("Nuclei Segmentation")
        sc.pl.spatial(self.sp_adata, color=["cell_count"], img_key=None, frameon=False, ax=axes[2],title='')
        _ = axes[2].set_title("Cell Count")
        plt.savefig(os.path.join(self.out_dir, 'nuclei_segmentation.png'))
        plt.close()
        
        self.sp_adata.uns['cell_locations'] = self.DissectSegRes(self.sp_adata.obsm['image_features'])
        self.sp_adata.obsm['image_features']['segmentation_label'] = (
            self.sp_adata.obsm['image_features']['segmentation_label'].astype(int)
        )
        self.sp_adata.obsm['image_features']['segmentation_centroid'] = (
            self.sp_adata.obsm['image_features']['segmentation_centroid'].astype(str)
        )  
        self.sp_adata.write_h5ad(os.path.join(self.out_dir, 'sp_adata_ns.h5ad'))

            coord_cell = self.sp_adata.uns['cell_locations'].dropna()
            coord_cell.columns = ['pxl_row_in_fullres', 'pxl_col_in_fullres', 'spot_index', 'cell_index', 'cell_nums']
            coord_cell.to_csv(os.path.join(self.out_dir, "position_all_tissue_sc.csv"))
        else:
            fig, axes = plt.subplots(1,2,figsize=(14,6),dpi=300)
            self.image.show("image", ax=axes[0])
            _ = axes[0].set_title("H&E")
            self.image.show("segmented_stardist_default", cmap="jet", interpolation="none", ax=axes[1])
            _ = axes[1].set_title("Nuclei Segmentation")
            plt.savefig(os.path.join(self.out_dir, 'nuclei_segmentation.png'))
            plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simulation sour_sep')
    parser.add_argument('--tissue', type=str, help='tissue name', default=None)
    parser.add_argument('--out_dir', type=str, help='output path', default=None)

    ## my add
    parser.add_argument('--roi_path', type=str, help='ROI path', default=None)
    parser.add_argument('--adata_path', type=str, help='ST data path (optional)', default=None)
    parser.add_argument('--img_path', type=str, help='H&E stained image data path', required=True)
    
    # parser.add_argument('--ST_Data', type=str, help='ST data path', default=None)
    # parser.add_argument('--Img_Data', type=str, help='H&E stained image data path', default=None)
    parser.add_argument(
        '--prob_thresh', 
        type=float, 
        help='object probability threshold, decrease this parameter if too many nucleus are missing', 
        default=0.5
    )    
    parser.add_argument('--max_cell_number', type=int, help='maximum cell number per spot', default=20)
    parser.add_argument('--min_counts', type=int, help='minimum UMI count per spot', default=500)
    parser.add_argument('--tilesize', type=int, help='Tile size for processing large images (default: auto-detect based on image size). Use smaller values (e.g., 2000) for very large images to avoid OOM errors.', default=None)
    args = parser.parse_args()
        
    NS = FineST_StarDist_nuclei(
        args.tissue, 
        args.out_dir, 

        ## my add
        args.roi_path,    
        args.img_path, 
        args.adata_path, 
        # args.ST_Data, 
        # args.Img_Data, 

        args.prob_thresh, 
        args.max_cell_number, 
        args.min_counts,
        args.tilesize
    )

    NS.NucleiSegmentation()


##########
# NPC
##########
# conda activate FineST
# python ./demo/StarDist_nuclei_segmente.py \
#     --tissue NPC_allspot_p075 \
#     --out_dir FineST_tutorial_data/NucleiSegments \
#     --adata_path FineST_tutorial_data/SaveData/adata_imput_all_spot.h5ad \
#     --img_path FineST_tutorial_data/20210809-C-AH4199551.tif \
#     --prob_thresh 0.75

##########
# CRC16 with ROI
##########
# python ./demo/StarDist_nuclei_segmente.py \
#     --tissue CRC16um_ROI_test \
#     --out_dir ./Dataset/CRC16um/StarDist/DataOutput \
#     --roi_path ./Dataset/CRC16um/ResultsROIs/ROI4_shape.csv \
#     --adata_path ./Dataset/CRC16um/Colon_Cancer_square_016um.h5ad \
#     --img_path ./Dataset/CRC16um/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
#     --prob_thresh 0.5 \
#     --max_cell_number 20 \
#     --min_counts 500