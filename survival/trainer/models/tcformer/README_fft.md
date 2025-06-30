
# SMILE: Self-sufficient Multiple Instance Learning with Token Labeling for Whole Slide Image Classification


## Abstract
Whole Slide Image(WSI) classification, a crucial component of digital pathology, confronts distinct challenges due to the vast and complex cellular structures it involves. Addressing the complexities of gigapixel-scale WSIs with only slide-level labels, Multiple Instance Learning(MIL) has proven pivotal in training method. Recent MIL algorithm attempting to improve WSI classification by using both slide-level and patch-level labels are hindered by the high cost of patch-level annotation. Our study introduces Self-sufficient Multiple Instance Learning with Token Labeling (SMILE), a novel method enhancing MIL for WSI classification. SMILE effectively employs Vision Transformers(ViT) and introduces a novel token labeling technique that generates low-cost, patch-level pseudo labels. This approach provides dense supervision, significantly improving the interpretation of output patch tokens. Our comprehensive experiments demonstrate that SMILE substantially boosts the performance of ViT-based models. By balancing the focus on both global and local image features, SMILE overcomes the limitations of previous ViT-based MIL methods, offering a more accurate and efficient solution for WSI classification in digital pathology.



## Usage
The following commands are examples of running the code for Camelyon16.

```bash
# Max-Pooling
CUDA_VISIBLE_DEVICES=3  python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code maxpool_nolinearlayer --bag_loss ce --model_type maxpool --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam  --scheduler --data_dir /longterm/dsmil_c16_pt

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code maxpool_nolinearlayer --bag_loss ce --model_type maxpool --log_data --split_dir tcga_dsmil --dataset tcga_dsmil --features imagenet  --opt adam  --scheduler --data_dir /longterm/dsmil_tcga_lung &


CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code maxpool_nolinearlayer --bag_loss ce --model_type maxpool --log_data --split_dir tcga_dsmil --dataset tcga_dsmil --features imagenet  --opt adam  --scheduler --data_dir /longterm/WSI/TCGA-ESCA 

# Mean-Pooling
CUDA_VISIBLE_DEVICES=1 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code meanpool_nolinearlayer --bag_loss ce --model_type meanpool --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam  --scheduler --data_dir /longterm/dsmil_c16_pt &

CUDA_VISIBLE_DEVICES=1 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code meanpool_nolinearlayer --bag_loss ce --model_type meanpool --log_data --split_dir tcga_dsmil --dataset tcga_dsmil --features imagenet  --opt adam  --scheduler --data_dir /longterm/dsmil_tcga_lung &

# AB-MIL
python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code abmil --bag_loss ce --model_type abmil --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam  --scheduler --data_dir  /longterm/WSI/TCGA-ESCA 

# CaiT
CUDA_VISIBLE_DEVICES=5 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code cait --bag_loss ce --model_type cait --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam  --scheduler --data_dir  /longterm/WSI/res_c16_pt

CUDA_VISIBLE_DEVICES=4 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code cait --bag_loss ce --model_type cait --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam  --scheduler --data_dir  /longterm/WSI/TCGA-ESCA 

# DeiT
CUDA_VISIBLE_DEVICES=4 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code deit --bag_loss ce --model_type deit --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam  --scheduler --data_dir  /longterm/WSI/TCGA-ESCA 

# DeepViT
CUDA_VISIBLE_DEVICES=1 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code deepvit --bag_loss ce --model_type deepvit --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam  --scheduler --data_dir  /longterm/WSI/TCGA-ESCA 

# SimpleViT
CUDA_VISIBLE_DEVICES=1,7 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code simplevit --bag_loss ce --model_type simplevit --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam  --scheduler --data_dir  /longterm/WSI/res_c16_pt

CUDA_VISIBLE_DEVICES=1 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code simplevit --bag_loss ce --model_type simplevit --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam  --scheduler --data_dir  /longterm/WSI/TCGA-ESCA 

# CRATE
CUDA_VISIBLE_DEVICES=2 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code crate --bag_loss ce --model_type crate --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam  --scheduler --data_dir  /longterm/WSI/TCGA-ESCA 

# DS-MIL
python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code dsmil --bag_loss ce --model_type dsmil --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam  --scheduler --data_dir {data patch} 

# CLAM(SB)
python main.py --drop_out --early_stopping --lr 2e-4 --k 5 --exp_code clam_sb --bag_loss ce --model_type clam_sb --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam  --scheduler --data_dir /longterm/WSI/res_c16_pt

# CLAM(MB)
python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --k_start 2 --exp_code clam_mb --bag_loss ce --model_type clam_mb --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam  --scheduler --data_dir /longterm/WSI/TCGA-ESCA

# TransMIL
python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code transmil --bag_loss ce --model_type transmil --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam --lookahead_opt  --scheduler --data_dir /longterm/WSI/dsmil_c16_pt 

# DTFD-MIL
CUDA_VISIBLE_DEVICES=5 python main.py --drop_out --early_stopping --lr 2e-4 --k 1 --k_start 0 --k_end 1 --exp_code dtfd_maxmins --bag_loss ce --model_type dtfd --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam --scheduler --data_dir /longterm/dsmil_c16_pt --input_dim 512 --min_epochs 80 --distill_type AFS &

CUDA_VISIBLE_DEVICES=3 python main.py --drop_out --early_stopping --lr 2e-4 --k 5 --exp_code dtfd --bag_loss ce --model_type dtfd --log_data --split_dir tcga_dsmil --dataset tcga_dsmil --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/dsmil_tcga_lung --input_dim 512 --min_epochs 150  --k_start 0 --k_end 1 --distill_type AFS

CUDA_VISIBLE_DEVICES=5 python main.py --drop_out --early_stopping --lr 2e-4 --k_start 1 --exp_code dtfd_AFS --bag_loss ce --model_type dtfd --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/TCGA-ESCA --input_dim 1024 --min_epochs 20 --distill_type AFS &

CUDA_VISIBLE_DEVICES=5 python main.py --drop_out --early_stopping --lr 2e-4 --k 1 --exp_code dtfd_MaxMinS --bag_loss ce --model_type dtfd --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/TCGA-ESCA --input_dim 1024 --min_epochs 20 --distill_type MaxMinS --k_end 

CUDA_VISIBLE_DEVICES=3 python main.py --drop_out --early_stopping --lr 2e-4 --k 5 --exp_code dtfd_AFS_auc --bag_loss ce --model_type dtfd --log_data --split_dir camelyon16_dtfd --dataset camelyon16 --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/dtfd_c16_pt --input_dim 1024 --min_epochs 20 --distill_type AFS --k_start 0 --k_end 1

CUDA_VISIBLE_DEVICES=1 python main.py --drop_out --early_stopping --lr 2e-4 --k 5 --exp_code dtfd_MaxMinS_auc --bag_loss ce --model_type dtfd --log_data --split_dir camelyon16_dtfd --dataset camelyon16 --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/dtfd_c16_pt --input_dim 1024 --min_epochs 20 --distill_type MaxMinS --k_start 3 --k_end 4

# MHIM (pretrain)
  # TransMIL backbone
CUDA_VISIBLE_DEVICES=7 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code mhim_tea --bag_loss ce --model_type mhim_pure --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam --scheduler --data_dir /longterm/dsmil_c16_pt --baseline=selfattn

CUDA_VISIBLE_DEVICES=3 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code mhim_tea --bag_loss ce --model_type mhim_pure --log_data --split_dir camelyon16_dtfd --dataset camelyon16 --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/dtfd_c16_pt --baseline=selfattn --k_start 4 --k_end 5

CUDA_VISIBLE_DEVICES=2 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code mhim_tea --bag_loss ce --model_type mhim_pure --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/TCGA-ESCA --baseline=selfattn --k_start 0 --k_end 1

CUDA_VISIBLE_DEVICES=6 python main.py --drop_out --early_stopping --lr 2e-4 --k 1 --k_start 2 --k_end 3  --exp_code mhim_tea --bag_loss ce --model_type mhim_pure --log_data --split_dir tcga_dsmil --dataset tcga_dsmil --features imagenet  --opt adam --scheduler --data_dir /longterm/dsmil_tcga_lung --baseline=selfattn &

# MHIM (student)
   # TransMIL backbone
CUDA_VISIBLE_DEVICES=7 python main.py --drop_out --early_stopping --lr 2e-4 --k 5 --exp_code mhim_stu --bag_loss ce --model_type mhim --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/dsmil_c16_pt --baseline=selfattn --init_stu_type=fc  --mask_ratio=0.1 --mask_ratio_l=0. --mask_ratio_h=0.01 --mask_ratio_hr=0.5 --teacher_init /home/beidiz/SMILE/results/camelyon16/imagenet/mhim_tea_s1/TransMIL_tea --max_epochs 0 --k_start 3

CUDA_VISIBLE_DEVICES=3 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --k_start 4 --k_end 5 --exp_code mhim_stu --bag_loss ce --model_type mhim --log_data --split_dir ESCA --dataset ESCA --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/TCGA-ESCA  --baseline=selfattn --init_stu_type=fc  --mask_ratio=0.6 --mask_ratio_l=0.2 --cl_alpha=0.5 --teacher_init /home/beidiz/SMILE/results/ESCA/imagenet/mhim_tea_s1/TransMIL_tea

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --k_start 3 --k_end 4 --exp_code mhim_stu --bag_loss ce --model_type mhim --log_data --split_dir tcga_dsmil --dataset tcga_dsmil --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/dsmil_tcga_lung  --baseline=selfattn --init_stu_type=fc  --mask_ratio=0.1 --mask_ratio_l=0. --mask_ratio_h=0.01 --mask_ratio_hr=0.5 --teacher_init /home/beidiz/SMILE/results/tcga_dsmil/imagenet/mhim_tea_s1/TransMIL_tea

CUDA_VISIBLE_DEVICES=1 python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --k_start 4 --k_end 5 --exp_code mhim_stu --bag_loss ce --model_type mhim --log_data --split_dir camelyon16_dtfd --dataset camelyon16 --features imagenet  --opt adam --scheduler --data_dir /longterm/WSI/dtfd_c16_pt --baseline=selfattn --init_stu_type=fc  --mask_ratio=0. --mask_ratio_l=0.8 --mask_ratio_h=0.03 --mask_ratio_hr=0.5 --cl_alpha=0.1 --teacher_init /home/beidiz/SMILE/results/camelyon16/imagenet/mhim_tea_s1/TransMIL_tea

# ViT
CUDA_VISIBLE_DEVICES=1 python -u main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code vit_fft_phase_ifft --bag_loss ce --model_type vit --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam --lookahead_opt  --scheduler --data_dir /longterm/WSI/res_c16_pt_phase  2>&1 |tee ./Record/vit_fft_phase_ifft.log &

### CLU
CUDA_VISIBLE_DEVICES=2 python -u main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code clu --bag_loss ce --model_type clu --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam --lookahead_opt  --scheduler --data_dir /longterm/WSI/res_c16_pt_phase  2>&1 |tee ./Record/clu.log &

# ViT(Token Labeler)
python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code vit_token_labeler --bag_loss ce --model_type vit_maxpool --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam --lookahead_opt  --scheduler --data_dir {data patch} 

# ViT(Student Model)
python main.py --drop_out --early_stopping --lr 2e-4 --k 5  --exp_code vit --bag_loss ce --model_type vit --log_data --split_dir camelyon16 --dataset camelyon16 --features imagenet  --opt adam --lookahead_opt  --scheduler --data_dir {data path} --token_label --tl_model_path {token labeler path}
```

### Parameters for dataset
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `dataset` | Dataset to use. Options: `camelyon16`, `tcga`,|
| `data_dir` | Directory path where a feature embedding (.pt) file exists. |
| `split_dir` | manually specify the set of splits to use. |
| `results_dir` | Results directory for saving experiment results.|


### Parameters for learning
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model_type` | MIL algorithm for training.  Options:  `maxpool`, `meanpool`, `clam_sb`, `clam_mb`, `transmil`, `abmil`, `dtfd`, `mhim`, `vit` , `vit_maxpool`|
| `lr` | Learning rate. |
| `reg` | Weight decay. |
| `k` | The number of folds |
| `scheduler` | Use of Cosine Scheduler |


### Parameters for SMILE
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `n_layers` | Number of encoder blocks in ViT, default = `2`. |
| `emb_dim` | embedding dimension, default = `384`. |
| `alpha` | hyperparameter about hard negative token labeling for token labeler, default = `1.0`. |
| `beta` | hyperparameter about token labeling for student model, default = `1.0`. |
| `negative_tl` | Hard negative token labeling mode for token labeler. |
| `token_label` | Token labeling mode for student model. |
| `tl_model_path` | Directory path where the token labeler model is stored. |

