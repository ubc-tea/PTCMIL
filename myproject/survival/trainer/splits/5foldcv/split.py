import pandas as pd
import os


def get_all_csv_file(path, save_path,dataset):
    for d in dataset:
        csv_path = path + d + '_overall_survival_k='
        folds = ['0']
        for f in folds:
            csv_file_train = csv_path + f + '/train.csv'
            csv_file_val = csv_path + f + '/val.csv'
            df_train = pd.read_csv(csv_file_train)
            df_val = pd.read_csv(csv_file_val)

        selected_columns_train = df_train[['case_id', 'slide_id', 'OncotreeCode', 'dss_survival_days', 'dss_censorship','disc_label']]
        selected_columns_val = df_val[['case_id', 'slide_id', 'OncotreeCode', 'dss_survival_days', 'dss_censorship','disc_label']]
        selected_columns_train['dss_censorship'] = pd.to_numeric(selected_columns_train['dss_censorship'], errors='coerce')
        selected_columns_val['dss_censorship'] = pd.to_numeric(selected_columns_val['dss_censorship'], errors='coerce')
        combined_df = pd.concat([selected_columns_train, selected_columns_val], axis=0, ignore_index=True)
    combined_df.to_csv(save_path + d.lower() + '_all_clean.csv', index=False)
    # selected_columns_val.to_csv(save_path + d.lower() + '_all_clean.csv',index=False)

def get_kfolds(path,save_path, dataset):
    for d in dataset:
        csv_path = path + d + '_overall_survival_k='
        folds = ['0', '1', '2', '3', '4']
        for f in folds:
            csv_file_train = csv_path + f + '/train.csv'
            csv_file_val = csv_path + f + '/test.csv'
            df_train = pd.read_csv(csv_file_train)
            df_val = pd.read_csv(csv_file_val)

            selected_columns_train = df_train[['case_id']].drop_duplicates().reset_index(drop=True)
            selected_columns_val = df_val[['case_id']].drop_duplicates().reset_index(drop=True)
  
            number_ranges = range(len(selected_columns_train))

            selected_columns_train.insert(0, 'number_ranges', number_ranges)
            save_path_dataset = save_path + '_' + d.lower()
            if not os.path.exists(save_path_dataset):
                os.makedirs(save_path_dataset)
            selected_columns_train.columns = ['', 'train']
            selected_columns_val.columns = ['val']
            concatenated_columns = pd.concat([selected_columns_train, selected_columns_val], axis=1)
            concatenated_columns.to_csv(save_path_dataset+'/splits_'+ f +'.csv', index=False)
            print("Duplicates removed and saved")


