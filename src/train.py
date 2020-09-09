#!/bin/python3

import os
import pickle
import numpy as np
import pandas as pd
import warnings
import random
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score

sys.path.append('..')
from model.baselines import CreateBaselines

def _load():
    source = open('data/expr_unirep_solubility_mar24.pkl', 'rb')

    X_train = pickle.load(source)
    X_valid = pickle.load(source)
    Y_train = pickle.load(source)
    Y_valid = pickle.load(source)

    source.close()
    return X_train, X_valid, Y_train, Y_valid

def _load_enz():
    train = pd.read_hdf('data/enz_dfs.h5', 'train')
    valid = pd.read_hdf('data/enz_dfs.h5', 'valid')
    test = pd.read_hdf('data/enz_dfs.h5', 'test')

    enz_data = pd.read_excel('data/data_combined_cleaned_hhm.xlsx', sheet_name='combined_fixed')
    enz_data.rename(columns={'Family': 'Enzyme Family', 'Category3': 'Enzyme'}, inplace=True)

    return train, valid, test, enz_data

def save_pickle(data, dest):
    with open(dest, 'wb') as d:
        pickle.dump(data, d)

def load_pickle(source):
    with open(source, 'rb') as s:
        data = pickle.load(s)
    return data

def train_baseline(X_train, X_valid, Y_train, Y_valid, input_cols, out_file=None, model_name='Random forest'):

    model = CreateBaselines( 
        X_train = X_train[input_cols],
        X_test = X_valid[input_cols],
        y_train = Y_train,
        y_test = Y_valid, 
        n_jobs = -1
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.create_baseline(model_name, evaluate=False)

        train_auc, valid_auc = None, None

        best_model = getattr(model, model.type2attr[model_name])

        if model.y_train.nunique() == 2:
            train_auc = roc_auc_score(model.y_train, best_model.predict_proba(model.X_train)[:,1])
        if model.y_test.nunique() == 2:
            valid_auc = roc_auc_score(model.y_test, best_model.predict_proba(model.X_test)[:,1])

        if out_file is not None: save_pickle(model, out_file)
    
    return model, train_auc, valid_auc  

def test_enz_seeds(col, n, out_file, model_name):

    results = {}
    train, valid, _, enz_data = _load_enz()
    y_col = 'is_expression_successful'

    input_cols = pd.DataFrame(train['unirep'].values.tolist()).columns.tolist()
    results['input_cols'] = input_cols

    enzymes = enz_data[col].unique()

    for enzyme in enzymes:
        print(enzyme)
        results[enzyme] = {}

        enz_train = train.loc[train[col] == enzyme, ].copy()
        enz_valid = valid.loc[valid[col] == enzyme, ].copy()

        Y_train, Y_valid = enz_train[y_col], enz_valid[y_col]
        X_train = pd.DataFrame(enz_train['unirep'].values.tolist())
        X_valid = pd.DataFrame(enz_valid['unirep'].values.tolist())
        
        if X_valid.shape[0] <= 5 or X_train.shape[0] <= 5:
            continue

        for i in range(n):
            random.seed()

            model, train_auc, valid_auc = train_baseline(
                X_train = X_train,
                X_valid = X_valid,
                Y_train = Y_train,
                Y_valid = Y_valid,
                input_cols = input_cols,
                model_name=model_name
            )

            results[enzyme][i] = {
                'model': model,
                'train_auc': train_auc,
                'valid_auc': valid_auc
            }

            #print("Enzyme={}, i={}: Train AUC: {:.3f}, valid AUC: {:.3f}".format(enzyme, i+1, train_auc, valid_auc))
    
        save_pickle(results, out_file)

def test_freq_seeds(col, n, out_file, model_name):

    # load data
    train = pd.read_csv('data/Copy of train3.csv', index_col=0)
    valid = pd.read_csv('data/Copy of valid3.csv', index_col=0)

    y_col = 'is_expression_successful'

    # get cols
    valid_aas = 'ARNDCQEGHILKMFPSTWYV'
    cols = [col + aa for aa in valid_aas]

    X_train = train[cols].copy()
    X_valid = valid[cols].copy()

    Y_train, Y_valid = train[y_col], valid[y_col]
    results = {}
    for i in range(n):
        random.seed()

        model, train_auc, valid_auc = train_baseline(
            X_train = X_train,
            X_valid = X_valid,
            Y_train = Y_train,
            Y_valid = Y_valid,
            input_cols=cols,
            model_name=model_name
        )

        results[i] = {
            'model': model,
            'train_auc': train_auc,
            'valid_auc': valid_auc
        }

        print("i={}: Train AUC: {:.3f}, valid AUC: {:.3f}".format(i+1, train_auc, valid_auc))
    
    save_pickle(results, out_file)

    try:
        train_aucs = np.array([v['train_auc'] for v in results.values()])
        valid_aucs = np.array([v['valid_auc'] for v in results.values()])

        train_avg_auc, train_std_auc = np.mean(train_aucs), np.std(train_aucs)
        valid_avg_auc, valid_std_auc = np.mean(valid_aucs), np.std(valid_aucs)

        print("Train AUC: {:.3f} (+/- {:.3f})".format(train_avg_auc, train_std_auc))
        print("Valid AUC: {:.3f} (+/- {:.3f})".format(valid_avg_auc, valid_std_auc))
    except:
        pass   


def test_freq_seeds(col, n, out_file, model_name):

    # load data
    train = pd.read_csv('data/Copy of train3.csv', index_col=0)
    valid = pd.read_csv('data/Copy of valid3.csv', index_col=0)

    y_col = 'is_expression_successful'

    # get cols
    valid_aas = 'ARNDCQEGHILKMFPSTWYV'
    cols = [col + aa for aa in valid_aas]

    X_train = train[cols].copy()
    X_valid = valid[cols].copy()

    Y_train, Y_valid = train[y_col], valid[y_col]
    results = {}
    for i in range(n):
        random.seed()

        model, train_auc, valid_auc = train_baseline(
            X_train = X_train,
            X_valid = X_valid,
            Y_train = Y_train,
            Y_valid = Y_valid,
            input_cols=cols,
            model_name=model_name
        )

        results[i] = {
            'model': model,
            'train_auc': train_auc,
            'valid_auc': valid_auc
        }

        print("i={}: Train AUC: {:.3f}, valid AUC: {:.3f}".format(i+1, train_auc, valid_auc))
    
    save_pickle(results, out_file)

    try:
        train_aucs = np.array([v['train_auc'] for v in results.values()])
        valid_aucs = np.array([v['valid_auc'] for v in results.values()])

        train_avg_auc, train_std_auc = np.mean(train_aucs), np.std(train_aucs)
        valid_avg_auc, valid_std_auc = np.mean(valid_aucs), np.std(valid_aucs)

        print("Train AUC: {:.3f} (+/- {:.3f})".format(train_avg_auc, train_std_auc))
        print("Valid AUC: {:.3f} (+/- {:.3f})".format(valid_avg_auc, valid_std_auc))
    except:
        pass   


def test_seeds(n, input_cols, out_file, model_name):
    
    X_train, X_valid, Y_train, Y_valid = _load()

    results = {}
    results['input_cols'] = input_cols

    for i in range(n):
        random.seed()

        model, train_auc, valid_auc = train_baseline(
            X_train=X_train,
            X_valid=X_valid,
            Y_train=Y_train,
            Y_valid=Y_valid,
            input_cols=input_cols,
            model_name=model_name
        )

        results[i] = {
            'model': model,
            'train_auc': train_auc,
            'valid_auc': valid_auc
        }

        print("i={}: Train AUC: {:.3f}, valid AUC: {:.3f}".format(i+1, train_auc, valid_auc))

    save_pickle(results, out_file)

    try:
        train_aucs = np.array([v['train_auc'] for v in results.values()])
        valid_aucs = np.array([v['valid_auc'] for v in results.values()])

        train_avg_auc, train_std_auc = np.mean(train_aucs), np.std(train_aucs)
        valid_avg_auc, valid_std_auc = np.mean(valid_aucs), np.std(valid_aucs)

        print("Train AUC: {:.3f} (+/- {:.3f})".format(train_avg_auc, train_std_auc))
        print("Valid AUC: {:.3f} (+/- {:.3f})".format(valid_avg_auc, valid_std_auc))
    except:
        pass    

if __name__ == "__main__":
    y_col = 'is_expression_successful'
    unit_cols = np.arange(0, 1900, 1)

    solubility_cols = [
        'molecular_weight','aromaticity', 'instability index',
        'isoelectric point', 'gravy', 'flexibility',
        'solubility', 'helix', 'turn', 'sheet'
    ] 

    featureImps = pd.read_csv('UniRep_features_imp.csv')

    models = {
        'AAFreqSVM': 'aa_',
        'AAFreqLR': 'aa_',
        'AAFreqRF': 'aa_',
        'UniRepLR': unit_cols,
        'UniRepSVM': unit_cols,
        'UniRepRF': unit_cols,

    }

    path = os.path.realpath(os.path.join('data', 'seed_training'))
    os.makedirs(path, exist_ok=True)
    for k in sorted(models, key=lambda k: len(models[k]), reverse=False):
        out_file = os.path.join(path, k + '.pkl')
        input_cols = models[k]

        if k.endswith('RF'):
            model = 'Random forest'
        elif k.endswith('SVM'):
            model = 'Linear'
        elif k.endswith('LR'):
            model = 'Logistic'

        print(k, ':', model)

        if 'Enzyme' in k:
            test_enz_seeds(col=input_cols, n=10, out_file=out_file, model_name=model)
        elif 'AAFreq' in k:
            test_freq_seeds(col=input_cols, n=10, out_file=out_file, model_name=model)
        else:
            test_seeds(n=10, input_cols=input_cols, out_file=out_file, model_name=model)
