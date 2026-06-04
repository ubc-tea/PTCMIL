# Camelyon16 example

This folder contains two CAMELYON16 WSIs and matching intermediate feature files
for a small smoke test:

```text
example/Camelyon16/
  slides/
    normal_001.tif
    tumor_038.tif
  out/h5_files/
    normal_001.h5
    tumor_038.h5
  UNI_c16_pt/pt_files/
    normal_001.pt
    tumor_038.pt
  checkpoints/
    camelyon16_uni_ptcmil_clu7_best.pt
  dataset_csv/camelyon16_example.csv
  manifest.csv
```

## Files

- `slides/*.tif`: raw CAMELYON16 whole-slide images downloaded from the public
  PathML/CAMELYON16 Google Drive mirror:
  `https://drive.google.com/drive/folders/0BzsdkU4jWx9Ba2x1NTZhdzQ5Zjg?resourcekey=0-g2TRih6YKi5P2O1SiBB1LA`.
- `out/h5_files/*.h5`: copied from `/bigdata/beidi/Camelyon16/out/h5_files`.
  These HDF5 files contain both `features` and `coords`.
- `UNI_c16_pt/pt_files/*.pt`: copied from
  `/bigdata/WSI/Camelyon16/UNI_c16_pt/pt_files`. These are the UNI feature
  tensors that PTCMIL reads for MIL training/inference.
- `checkpoints/camelyon16_uni_ptcmil_clu7_best.pt`: recommended Camelyon16
  UNI PTCMIL checkpoint for smoke-test inference. It was selected from
  `local_vit_prompt_clu_UNI_adam_alpha01_moving_momentum1_normassn_hard_clu7_s1`
  fold 2 (`test_auc=0.99921875`, `test_acc=0.984375`).

## Re-download WSIs

The WSI files are already present in `slides/`. If they need to be restored, run:

```bash
cd /data/beidiz/published/PTCMIL
python example/Camelyon16/download_wsi.py
```

The script uses public Google Drive file ids parsed from the PathML tutorial data
folder and handles the Google Drive large-file confirmation page.

Reference page for the public tutorial data:
`https://www.medrxiv.org/content/10.1101/2021.07.07.21260138.full`.

## Example feature path

For PTCMIL code that expects a feature directory containing `pt_files/`, use:

```text
/data/beidiz/published/PTCMIL/example/Camelyon16/UNI_c16_pt
```

## Example checkpoint

Use this checkpoint for Camelyon16 UNI inference:

```text
/data/beidiz/published/PTCMIL/example/Camelyon16/checkpoints/camelyon16_uni_ptcmil_clu7_best.pt
```

It expects `model_type=vit_pmt_clu`, `input_dim=1024`, `emb_dim=384`,
`cluster_number=7`, and `n_classes=2`.

## Test inference

Run this block from the repository root to load the example checkpoint, read the
UNI feature tensors, and print slide-level labels:

```bash
cd /data/beidiz/published/PTCMIL
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from pathlib import Path
from types import SimpleNamespace
import sys
import torch

repo = Path.cwd()
sys.path.insert(0, str(repo / "classification"))

from models.model_vit_pmt_clu import ViT_PMT_CLU

if not torch.cuda.is_available():
    raise RuntimeError("ViT_PMT_CLU uses CUDA tensors in forward(); run this test with a GPU.")

device = torch.device("cuda")
checkpoint_path = repo / "example/Camelyon16/checkpoints/camelyon16_uni_ptcmil_clu7_best.pt"
feature_dir = repo / "example/Camelyon16/UNI_c16_pt/pt_files"

args = SimpleNamespace(
    input_dim=1024,
    emb_dim=384,
    cluster_number=7,
    n_layers=2,
    moving_pmt=True,
    momentum=1.0,
    merge_token=True,
)

model = ViT_PMT_CLU(n_classes=2, args=args).to(device)
state_dict = torch.load(checkpoint_path, map_location=device)
load_info = model.load_state_dict(state_dict, strict=False)

# The checkpoint was saved from a training model that included an unused encoder
# module. The current inference model does not use those encoder weights.
unexpected = [key for key in load_info.unexpected_keys if not key.startswith("encoder.")]
if load_info.missing_keys or unexpected:
    raise RuntimeError(f"Checkpoint mismatch: missing={load_info.missing_keys}, unexpected={unexpected}")

label_names = {0: "normal", 1: "tumor"}
model.eval()

with torch.no_grad():
    for feature_path in sorted(feature_dir.glob("*.pt")):
        features = torch.load(feature_path, map_location=device).float()
        _, probs, y_hat, _, _ = model(features)
        pred = int(y_hat.item())
        print(
            f"{feature_path.name}: "
            f"pred_label={pred} ({label_names[pred]}), "
            f"prob_normal={probs[0, 0].item():.6f}, "
            f"prob_tumor={probs[0, 1].item():.6f}"
        )
PY
```

Expected output:

```text
normal_001.pt: pred_label=0 (normal), prob_normal=0.997584, prob_tumor=0.002416
tumor_038.pt: pred_label=1 (tumor), prob_normal=0.000029, prob_tumor=0.999971
```
