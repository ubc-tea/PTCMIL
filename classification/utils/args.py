import argparse


def args_parser():
    # Generic training settings
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--dataset', type=str, default=None, choices=["camelyon16", 'tcga','PANDA'])
    parser.add_argument('--features', type=str, default=None, choices=['imagenet','dasmil','ctranspath','UNI','resnet50'])
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--log_iter', default=20, type=int, help='Log Frequency')
    parser.add_argument('--data_dir', type=str, default=None, help='data directory')
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--min_epochs', type=int, default=20, help='minimum number of epochs to train (default: 20)')
    parser.add_argument('--patience', type=int, default=10, help='patience of early stopping')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate (default: 0.0002)')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--results_dir', default=None, help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use')
    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd', 'radam', 'adamw'], default='adam')
    parser.add_argument('--lookahead_opt', action='store_true', default=False, help='lookahead optimizer')
    parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')
    parser.add_argument('--task', type=str, choices=['survival','classification'], default='classification', help='task to perform (default: classification)')
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'nllloss'], default='ce', help='slide-level classification loss function (default: ce)')
    parser.add_argument('--alpha_surv', type=float, default=0.0, help='weight given to uncensored patients')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'maxpool', 'meanpool', 'vit', 'transmil',  "dsmil", "abmil",'ilra','vit_pmt_clu'], help='type of model')
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')

    ### For CLAM 
    parser.add_argument('--no_inst_cluster', action='store_true', default=False, help='disable instance-level clustering')
    parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None, help='instance-level clustering loss function (default: None)')
    parser.add_argument('--subtyping', action='store_true', default=False, help='subtyping problem')
    parser.add_argument('--bag_weight', type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')


    ### For ViT
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--emb_dim', type=int, default=384)

    ### For DTFD
    parser.add_argument('--numGroup', default=5, type=int)
    parser.add_argument('--mDim', default=512, type=int)
    parser.add_argument('--numLayer_Res', default=0, type=int)
    parser.add_argument('--distill_type', choices=['MaxMinS', 'MaxS', 'AFS'],default='AFS', type=str)
    parser.add_argument('--droprate', default='0', type=float)
    parser.add_argument('--droprate_2', default='0', type=float)

    ## clustering
    parser.add_argument('--temp', type=float, default=0.07,help='temperature for loss function')
    parser.add_argument('--pmt_clu', action='store_true', default=False)
    parser.add_argument('--merge_token', action='store_true', default=False)
    parser.add_argument('--moving_pmt', action='store_true', default=False)
    parser.add_argument('--momentum', type=float, default=0.01)
    parser.add_argument('--cluster_number', default=5, type=int)

    ## Survival
    parser.add_argument('--n_bins', default=4, type=int, help='Number of bins of survival prediction')
    parser.add_argument('--lambda_reg', default=1e-4, type=float, help='L1-Regularization Strength (Default 1e-4)')

    ## fewshot
    parser.add_argument('--fewshot_path', default=None, type=str, help='pretrained model for few-shot learning')

    ## mambamil

    parser.add_argument('--mambamil_rate',type=int, default=5, help='mambamil_rate')
    parser.add_argument('--mambamil_layer',type=int, default=2, help='mambamil_layer')
    parser.add_argument('--mambamil_type',type=str, default='SRMamba', choices= ['Mamba', 'BiMamba', 'SRMamba'], help='mambamil_type')

    return parser.parse_args()
