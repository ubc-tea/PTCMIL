
# PTCMIL: Multiple Instance Learning via Prompt Token Clustering for Whole Slide Image Analysis


## Abstract
Multiple Instance Learning (MIL) has advanced WSI analysis but struggles with the complexity and heterogeneity of WSIs. Existing MIL methods face challenges in aggregating diverse patch information into robust WSI representations. While ViTs and clustering-based approaches show promise, they are computationally intensive and fail to capture task-specific and slide-specific variability. To address these issues, we propose PTCMIL, a novel Prompt Token Clustering-based ViT for MIL aggregation. By introducing learnable prompt tokens into the ViT backbone, PTCMIL unifies clustering and prediction tasks in an end-to-end manner. It dynamically aligns clustering with downstream tasks, using projection-based clustering tailored to each WSI, reducing complexity while preserving patch heterogeneity. Through token merging and prototype-based pooling, PTCMIL efficiently captures task-relevant patterns. Extensive experiments on eight datasets demonstrate its superior performance in classification and survival analysis tasks, outperforming state-of-the-art methods. Systematic ablation studies confirm its robustness and strong interpretability.

### Setting up the Conda Environment
To recreate the environment, run:
```bash
conda env create -f environment.yml
```

## Classification
We should enter the folder 
```bash
cd classification
```
We assume that the features are already extracted. We can choose to use feature extracted from ResNet50, CTransPath or UNI. The location of feature is stored at PATH_TO_FEATURE.
```bash

### Camelyon16
CUDA_VISIBLE_DEVICES=0 python main_classification.py --drop_out --early_stopping --lr 2e-4 --k 5 --exp_code ptcmil --bag_loss ce --model_type vit_pmt_clu --log_data --split_dir camelyon16 --dataset camelyon16 --features UNI  --opt adam --scheduler --data_dir PATH_TO_FEATURE --pmt_clu --alpha 0.1 --moving_pmt --momentum 0.1 --merge_token --input_dim 1024 --cluster_number 7

### TCGA-NSCLC
CUDA_VISIBLE_DEVICES=0 python main_classification.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code ptcmil --bag_loss ce --model_type vit_pmt_clu --log_data --split_dir tcga --dataset tcga --features UNI  --opt adam --scheduler --data_dir PATH_TO_FEATURE --pmt_clu --alpha 0.1 --moving_pmt --momentum 0.1 --merge_token --input_dim 1024 --cluster_number 5  

### PANDA
CUDA_VISIBLE_DEVICES=0 python -u main_classification.py --drop_out --early_stopping --lr 2e-4 --k 1 --exp_code ptcmil --bag_loss ce --model_type vit_pmt_clu --log_data --split_dir PANDA --dataset PANDA --features UNI --opt adam --scheduler --data_dir PATH_TO_FEATURE --pmt_clu --alpha 0.1 --moving_pmt --momentum 0.1 --merge_token --input_dim 1024 --n_classes 6 --cluster_number 5 


```
## Survival Prediction
We should enter the folder 
```bash
cd survival
```
The code of running on four datasets.
```bash
### BLCA
CUDA_VISIBLE_DEVICES=0 python main_survival.py --data_root_dir PATH_TO_FEATURE --split_dir tcga_blca --model_type ptcmil --mode path --which_splits 5foldcv --reg 1e-5 --lr 2e-4 --merge_token --moving_pmt --momentum 0.1 --cluster_number 5  --beta 0.1 --pmt_clu

### BRCA
CUDA_VISIBLE_DEVICES=0 python main_survival.py --data_root_dir PATH_TO_FEATURE --split_dir tcga_brca --model_type ptcmil --mode path --which_splits 5foldcv --reg 1e-5 --lr 2e-4 --merge_token --moving_pmt --momentum 0.1 --cluster_number 5  --beta 0.1 --pmt_clu

### CRC
CUDA_VISIBLE_DEVICES=0 python main_survival.py --data_root_dir PATH_TO_FEATURE --split_dir tcga_crc --model_type ptcmil --mode path --which_splits 5foldcv --reg 1e-5 --lr 2e-4 --merge_token --moving_pmt --momentum 0.1 --cluster_number 5  --beta 0.1 --pmt_clu

### LUAD
CUDA_VISIBLE_DEVICES=0 python main_survival.py --data_root_dir PATH_TO_FEATURE --split_dir tcga_luad --model_type ptcmil --mode path --which_splits 5foldcv --reg 1e-5 --lr 2e-4 --merge_token --moving_pmt --momentum 0.1 --cluster_number 5  --beta 0.1 --pmt_clu


```

### Parameters for dataset
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `dataset` | Dataset to use. Options: `camelyon16`, `tcga`, `PANDA`|
| `data_dir` | Directory path where a feature embedding (.pt) file exists. |
| `split_dir` | manually specify the set of splits to use. |
| `results_dir` | Results directory for saving experiment results.|


### Parameters for PTCMIL
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model_type` | MIL algorithm for training.  Options:  `maxpool`, `meanpool`, `clam_sb`, `clam_mb`, `transmil`, `abmil`, `dtfd`, `ilra`, `ptcmil` (For MambaMIL, PANTHER, DGR-MIL, we use the original repo to reproduce them.)|
| `alpha` | PTC loss term for classificsation. Default:0.1. |
| `beta` | PTC loss term for survival prediction. Default:0.1. |
| `momentum` | Decay factor for updating the prompt tokens. Default:0.1. |
| `lr` | Learning rate. Default:2e-4. |
| `k` | The number of folds. Default:5. |
| `scheduler` | Use of Cosine Scheduler |



