# WSI preprocessing

This folder vendors the WSI preprocessing pipeline used before PTCMIL
training (based on CLAM):

1. `create_patches_fp.py`: segment WSIs, create patch coordinate HDF5 files, and
   optionally stitch preview images.
2. `extract_features_fp_UNI.py`: read the generated patch coordinates and extract
   UNI features into `h5_files/` and `pt_files/`.

The copied CLAM dependencies live in this folder so the scripts can be run from
the PTCMIL root without setting `PYTHONPATH`.

## Camelyon16 example

Run from the repository root:

```bash
cd /data/beidiz/published/PTCMIL
OUT_DIR=preprocessing/outputs/Camelyon16

CUDA_VISIBLE_DEVICES=0 python preprocessing/create_patches_fp.py \
  --source /bigdata/Camelyon16/images \
  --save_dir "$OUT_DIR" \
  --patch_size 512 \
  --step_size 512 \
  --preset bwh_biopsy.csv \
  --seg \
  --patch \
  --stitch

CUDA_VISIBLE_DEVICES=0 python preprocessing/extract_features_fp_UNI.py \
  --data_h5_dir "$OUT_DIR" \
  --data_slide_dir /bigdata/Camelyon16/images \
  --csv_path "$OUT_DIR/process_list_autogen.csv" \
  --feat_dir "$OUT_DIR" \
  --batch_size 256 \
  --slide_ext .tif \
  --custom_downsample 2 
```

Expected output layout:

```text
/data/beidiz/published/PTCMIL/preprocessing/outputs/Camelyon16/
  process_list_autogen.csv
  patches/
  masks/
  stitches/
  h5_files/
  pt_files/
```

`--preset bwh_biopsy.csv` is resolved relative to `preprocessing/presets/`.
