import argparse
def args_parser():
    # Generic training settings
    parser = argparse.ArgumentParser(
    description='Configurations for Survival Analysis on TCGA Data.')

    ### Checkpoint + Misc. Pathing Parameters
    parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir',
                        help='Data directory to WSI features (extracted via CLAM')
    parser.add_argument('--seed', 			 type=int, default=1,
                        help='Random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', 			     type=int, default=5,
                        help='Number of folds (default: 5)')
    parser.add_argument('--k_start',		 type=int, default=-1,
                        help='Start fold (Default: -1, last fold)')
    parser.add_argument('--k_end',			 type=int, default=-1,
                        help='End fold (Default: -1, first fold)')
    parser.add_argument('--results_dir',     type=str, default='./results',
                        help='Results directory (Default: ./results)')
    parser.add_argument('--which_splits',    type=str, default='5foldcv',
                        help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
    parser.add_argument('--split_dir',       type=str, default='tcga_blca',
                        help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca)')
    parser.add_argument('--log_data',        action='store_true', 
                        help='Log data using tensorboard')
    parser.add_argument('--overwrite',     	 action='store_true', default=False,
                        help='Whether or not to overwrite experiments (if already ran)')
    parser.add_argument('--load_model',        action='store_true',
                        default=False, help='whether to load model')
    parser.add_argument('--path_load_model', type=str,
                        default='/path/to/load', help='path of ckpt for loading')
    parser.add_argument('--start_epoch',              type=int,
                        default=0, help='start_epoch.')

    ### Model Parameters.
    parser.add_argument('--model_type',      type=str, choices=['snn', 'amil', 'mcat', 'motcat','deepset','vit','ptcmil', 'clam', 'transmil','abmil','dsmil','ilra'], 
                        default='motcat', help='Type of model (Default: motcat)')
    parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'],
                        default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
    parser.add_argument('--fusion',          type=str, choices=['None', 'concat'], default='concat', help='Type of fusion. (Default: concat).')
    parser.add_argument('--apply_sig',		 action='store_true', default=False,
                        help='Use genomic features as signature embeddings.')
    parser.add_argument('--apply_sigfeats',  action='store_true',
                        default=False, help='Use genomic features as tabular features.')
    parser.add_argument('--drop_out',        action='store_true',
                        default=True, help='Enable dropout (p=0.25)')
    parser.add_argument('--model_size_wsi',  type=str,
                        default='small', help='Network size of AMIL model')
    parser.add_argument('--model_size_omic', type=str,
                        default='small', help='Network size of SNN model')

    ### Optimizer Parameters + Survival Loss Function
    parser.add_argument('--opt',             type=str,
                        choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--batch_size',      type=int, default=1,
                        help='Batch Size (Default: 1, due to varying bag sizes)')
    parser.add_argument('--gc',              type=int,
                        default=32, help='Gradient Accumulation Step.')
    parser.add_argument('--max_epochs',      type=int, default=20,
                        help='Maximum number of epochs to train (default: 20)')
    parser.add_argument('--lr',				 type=float, default=2e-4,
                        help='Learning rate (default: 0.0002)')
    parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv',
                        'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: nll_surv)')
    parser.add_argument('--label_frac',      type=float, default=1.0,
                        help='fraction of training labels (default: 1.0)')
    parser.add_argument('--bag_weight',      type=float, default=0.7,
                        help='clam: weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--reg', 			 type=float, default=1e-5,
                        help='L2-regularization weight decay (default: 1e-5)')
    parser.add_argument('--alpha_surv',      type=float, default=0.0,
                        help='How much to weigh uncensored patients')
    parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'],
                        default='None', help='Which network submodules to apply L1-Regularization (default: None)')
    parser.add_argument('--lambda_reg',      type=float, default=1e-4,
                        help='L1-Regularization Strength (Default 1e-4)')
    parser.add_argument('--weighted_sample', action='store_true',
                        default=True, help='Enable weighted sampling')
    parser.add_argument('--early_stopping',  action='store_true',
                        default=False, help='Enable early stopping')

    ### ViT Parameters
    parser.add_argument('--emb_dim',type=float, default=384)
    parser.add_argument('--n_layers',type=int, default=2)

    ### PTCMIL Parameters
    parser.add_argument('--cluster_number',type=int, default=5)
    parser.add_argument('--moving_pmt', action='store_true', default=False)
    parser.add_argument('--pmt_clu', action='store_true', default=False)
    parser.add_argument('--merge_token', action='store_true', default=True)
    parser.add_argument('--beta', 			 type=float, default=0.2)
    parser.add_argument('--target_col', type=str, default='os_survival_days')
    parser.add_argument('--data_source', type=str, default=None, help='manually specify the data source')
    parser.add_argument('--n_label_bins', type=int, default=4, help='number of bins for event time discretization')
    parser.add_argument('--bag_size', type=int, default=-1)
    parser.add_argument('--train_bag_size', type=int, default=-1)
    parser.add_argument('--val_bag_size', type=int, default=-1)
    parser.add_argument('--split_names', type=str, default='train,val',
                        help='delimited list for specifying names within each split')
    parser.add_argument('--num_workers', type=int, default=2)
    args, unknown = parser.parse_known_args()
    return args
