import os
import logging
import datetime as dt
import pandas as pd
import numpy as np
import argparse
import get_data as gt

from labeling_functions import lfs_dict

from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model.label_model import LabelModel

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Weak Supervision Classifier")
    parser.add_argument('--name_file', type=str, default = '/data', help='general name of data files')
    parser.add_argument('--dev_bool', type=bool, default=True, help='dev split included?')
    parser.add_argument('--tie_break_policy', type=str, default='abstain', help='Tie break policy for predicting labels (random vs abstain)')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

### Global variables
format_time_input = '%d/%m/%y'

### Helper functions
def check_dir_exists(path):
    """Checks if folder directory already exists, else makes directory.
    Args:
        path (str): folder path for saving.
    """
    is_exist = os.path.exists(path)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"Creating {path} folder")
    else:
        print(f"Folder exists: {path}")

def load_split(data_dir, name_file, split):

    if split=='unlabeled':
        df = gt.get_tweets('mps')
    else:
        df = pd.read_csv(data_dir+name_file+'_'+split+'.csv')
    return df

def get_metrics(L, Y, label_model, tie_break_policy, metrics, split):
    print(metrics)
    scores = label_model.score(L=L, Y=Y, metrics=metrics, tie_break_policy=tie_break_policy)
    # for metric in metrics:
    #     # print(f"Label Model {metric} for {split} set: {scores[metric] * 100:.1f}")
    # print('\n')
    
def save_results(df, path):
    df.to_csv(path, index=False)

map_time = lambda x: dt.datetime.strptime(x, format_time_input)
    
def get_max_date_str(df: pd.DataFrame, 
                     format_output: str = '%y%m%d', 
                     column: str = 'date') -> str:

    dates = df[column].apply(map_time).values
    max_date = pd.to_datetime(max(dates))
    # max_date_str = max_date.strftime(max_date, format_output)
    max_date_str = max_date.strftime(format_output)

    return max_date_str

def main(data_dir: str,
         output_dir: str,
         name_file: str,
         splits: list,
         tie_break_policy: str,
         kwargs_fit: dict,
         metrics: list,
         lfs_dict: dict,
         ):

    ### Time 
    format_output = '%y%m%d'
    now = dt.datetime.now()
    datetime_str = str(now)
    date_str = now.strftime(format_output)

    ### Setup logging
    logger.setLevel(logging.DEBUG)
    log_dir = f'{output_dir}/logs'
    check_dir_exists(log_dir)
    handler = logging.FileHandler(f"{log_dir}/{datetime_str}.log")
    logger.addHandler(handler)

    logger.info(f"--Start: {datetime_str}--")

    ### Load data
    dataset = dict()
    split = 'train'
    dataset[split] = load_split(data_dir, name_file, split)

    split = 'test'
    dataset[split] = load_split(data_dir, name_file, split)

    if 'dev' in splits:
        split = 'dev'
        dataset[split] = load_split(data_dir, name_file, split)

    n_classes = []
    for split in splits:
        n = len(dataset[split]['label'].unique())
        n_classes.append(n)

    if len(np.unique(np.array(n_classes))) == 1:
        cardinality = n_classes[0]
    else:
        raise Exception('Not same number of categories in splits')

    for split in splits:
        logger.info(f'--{len(dataset[split])} examples in {split} set--\n')
        logger.info(f"--label distribution for {split} set--\n{dataset[split]['label'].value_counts()}")

    split = 'unlabeled'
    dataset[split] = load_split(data_dir, name_file, split)
    logger.info(f'--{len(dataset[split])} examples in {split} set--\n')


    max_date_str = get_max_date_str(dataset[split])

    ### Labeling functions

    labeling_functions = lfs_dict

    ### Train model
    logger.info("--Model Training--")
    lfs = [item[1] for item in labeling_functions.items()]

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=dataset['train'])
    L_test = applier.apply(df=dataset['test'])
    L_unlabeled = applier.apply(df=dataset['unlabeled'])

    label_model = LabelModel(cardinality=cardinality, verbose=False)
    Y_test = dataset['test'].label.values

    if 'dev' in splits:
        L_dev = applier.apply(df=dataset['dev'])
        Y_dev = dataset['dev'].label.values
        start = dt.datetime.now()
        label_model.fit(L_train=L_train, Y_dev=Y_dev, **kwargs_fit)
    else:
        start = dt.datetime.now()
        label_model.fit(L_train=L_train, **kwargs_fit)

    ### Model evaluation
    logger.info("--Model Evaluation--")
    test_preds = label_model.predict(L_test, tie_break_policy=tie_break_policy)

    split = 'test'
    get_metrics(L_test, Y_test, label_model, tie_break_policy, metrics, split)

    split = 'dev'
    if split in splits:
        dev_preds = label_model.predict(L_dev, tie_break_policy=tie_break_policy)
        get_metrics(L_dev, Y_dev, label_model, tie_break_policy, metrics, split)

    ### Labeling unseen data
    logger.info("--Prediction of unseen data--")
    df_unlabeled = dataset['unlabeled'].copy()
    predicted_labels = label_model.predict(L_unlabeled, tie_break_policy=tie_break_policy)
    df_unlabeled.loc[:, 'label'] = predicted_labels

    ### Save output
    logger.info("--Saving predicted labels--")
    output_name = 'data_'+date_str+'labeled_'+max_date_str+'covered.csv'
    filename = data_dir + output_name
    save_results(df_unlabeled, filename)

    end = dt.datetime.now()
    runtime = str(end-start)
    print(f'runtime: {runtime}')
    logger.info("--End--")

if __name__ == '__main__':
    args = parse_args()

    # Set dirs
    path = os.getcwd()
    main_dir = os.path.split(path)[0]
    data_dir = f"{main_dir}/data"
    output_dir = f'{main_dir}'

    if args.dev_bool:
        splits = ['train', 'test', 'dev']
    else:
        splits = ['train', 'test']
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_macro']
    kwargs_fit = {'n_epochs': 100, 'log_freq': 10, 'seed': 23}

    main(data_dir, output_dir, args.name_file, splits, args.tie_break_policy,
         kwargs_fit, metrics, lfs_dict)