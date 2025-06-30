from __future__ import print_function
import torch
import torch.utils
import torch.utils.cpp_extension
from timeit import default_timer as timer

import pdb
import os
import math
import sys

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from utils.args import args_parser
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

def main_classification(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        print(args.split_dir)
        print('fold: ', i)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)
        
        filename = os.path.join(args.results_dir, 'split_{}_summary.pkl'.format(i))
        save_pkl(filename, {"test_auc": test_auc, "test_acc": test_acc})

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

def main_survival(args):
	#### Create Results Directory
	if not os.path.isdir(args.results_dir):
		os.mkdir(args.results_dir)

	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end

	latest_val_cindex = []
	folds = np.arange(start, end)

	### Start 5-Fold CV Evaluation.
	for i in folds:
		start = timer()
		seed_torch(args.seed)
		results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
		if os.path.isfile(results_pkl_path):
			print("Skipping Split %d" % i)
			continue

		### Gets the Train + Val Dataset Loader.
		train_dataset, val_dataset = dataset.return_splits(from_id=False, 
				csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
		
		print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
		datasets = (train_dataset, val_dataset)

		### Run Train-Val on Survival Task.
		if args.task_type == 'survival':
			val_latest, cindex_latest = train(datasets, i, args)
			latest_val_cindex.append(cindex_latest)

		### Write Results for Each Split to PKL
		save_pkl(results_pkl_path, val_latest)
		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))

	### Finish 5-Fold CV Evaluation.
	if args.task_type == 'survival':
		results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})

	if len(folds) != args.k:
		save_name = 'summary_partial_{}_{}.csv'.format(start, end)
	else:
		save_name = 'summary.csv'

	results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))

def seed_torch(seed=1):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    args = args_parser()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)
    print(args)
    if args.task =='classification':
        print('\nLoad Dataset')
        if args.dataset == 'camelyon16':
            dataset = Generic_MIL_Dataset(csv_path ='dataset_csv/camelyon16_total.csv',
                                    data_dir=args.data_dir,
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'normal':0, 'tumor':1},
                                    patient_strat=False,
                                    ignore=[])
        if args.dataset == 'tcga':
            dataset = Generic_MIL_Dataset(csv_path ='dataset_csv/TCGA_total.csv',
                                    data_dir=args.data_dir,
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'LUAD':0, 'LUSC':1},
                                    patient_strat=False,
                                    ignore=[])
        if args.dataset == 'PANDA':
                dataset = Generic_MIL_Dataset(csv_path ='dataset_csv/PANDA_all.csv',
                                    data_dir=args.data_dir,
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'0':0, "1":1, '2':2, '3':3, '4':4, '5':5},
                                    patient_strat=False,
                                    ignore=[])

        if args.results_dir is None:
            args.results_dir = "./results/{}/{}".format(args.dataset, args.features)

        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir, exist_ok=True)

        args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
        if not os.path.isdir(args.results_dir):
            os.mkdir(args.results_dir)

        args.split_dir = os.path.join('splits', args.split_dir)

        print('split_dir: ', args.split_dir)
        assert os.path.isdir(args.split_dir)

        results = main_classification(args)
        print("finished!")
        print("end script")
    
        
        ### Creates results_dir Directory.
        if not os.path.isdir(args.results_dir):
            os.mkdir(args.results_dir)

        ### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
        args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed))
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
            print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
            sys.exit()

        ### Sets the absolute path of split_dir
        args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
        print("split_dir", args.split_dir)
        assert os.path.isdir(args.split_dir)
        settings.update({'split_dir': args.split_dir})

        with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
            print(settings, file=f)
        f.close()

        print("################# Settings ###################")
        for key, val in settings.items():
            print("{}:  {}".format(key, val))   
        results = main_classification(args)
        print("finished!")
        print("end script")
      
                



