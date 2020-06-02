import random as rnd

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import os

def mkdir_tree(source):
    if source is None:
        source = 'default'
    base_dirs = ['../data/clf_meta/%s/'%source]

    # hostname = socket.gethostname()
    # print('hostname is', hostname)
    # if 'arc-ts.umich.edu' in hostname:
    #     base_dirs.append('/scratch/cbudak_root/cbudak/lbozarth/fakenews/data/clf_meta/%s'%source)
    print('base_dirsssssss', base_dirs)
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print('mkdir', base_dir)
            os.mkdir(base_dir)

        if source == 'RDEL':
            subdirs = ['models', 'preds', 'features', 'vectorizers']
        else:
            subdirs = ['models', 'preds']

        datasets = ['default', 'events', 'increment', 'nela', 'fakenewscorpus', 'forecast', 'events_v2', 'valarch']
        datasets2 = ['default', 'nela', 'fakenewscorpus']
        datasets3 = ['bydomains', 'byforecast', 'basic']
        for d in subdirs:
            sub_dir = os.path.join(base_dir, d)
            if not os.path.exists(sub_dir):
                print('mkdir', sub_dir)
                os.mkdir(sub_dir)
            for dataset in datasets:
                dataset_path = os.path.join(sub_dir, dataset)
                if not os.path.exists(dataset_path):
                    print('mkdir', dataset_path)
                    os.mkdir(dataset_path)
                if dataset == 'increment' or dataset=='valarch':
                    for ds2 in datasets2:
                        ds2_path = os.path.join(dataset_path, ds2)
                        if not os.path.exists(ds2_path):
                            print('mkdir', ds2_path)
                            os.mkdir(ds2_path)
                        if dataset=='valarch':
                            for ds3 in datasets3:
                                ds3_path = os.path.join(ds2_path, ds3)
                                if not os.path.exists(ds3_path):
                                    print('mkdir', ds3_path)
                                    os.mkdir(ds3_path)

    print('finished making directory tree')
    return

def gen_rand_dates(dataset="default", start_date='2016-06-15', end_date='2016-12-30', n=10):
    rand_dates = []
    for i in range(n):
        dt = pd.to_datetime(rnd.choice(pd.bdate_range(start_date, end_date)))
        rand_dates.append(dt)
    print(sorted(rand_dates))
    return rand_dates

def evaluate_clf_preformance(y_true, y_pred, y_pred_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc = roc_auc_score(y_true, y_pred_prob)
    accu = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', pos_label=1)
    f1_micro = f1_score(y_true, y_pred, average='micro', pos_label=1)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', pos_label=1)
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    f1s = f1_score(y_true, y_pred, average=None)
    f1_real = None
    for f in f1s:
        if f!=f1_pos:
            f1_real = f
    print('auc, accuracy, f1_micro, f1_macro, f1_weighted, f1_fake, f1_real are', auc, accu, f1_micro, f1_macro, f1_weighted, f1_pos, f1_real)
    return {'auc_score':auc, 'accuracy_score':accu, 'f1_micro':f1_micro, 'f1_macro':f1_macro, 'f1_weighted':f1_weighted,
            'f1_fake':f1_pos, 'f1_real':f1_real, 'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp}

import numpy as np
def gen_precision_recall(y_true, y_pred):
    percision, recall, f1s, _ = precision_recall_fscore_support(y_true, y_pred, average=None, pos_label=1)
    if len(f1s) == 1:
        return np.nan, np.nan  # [1][1]; too few values
    return percision[1], recall[1] #for fake only

def gen_fnr_fpr(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # # Sensitivity, hit rate, recall, or true positive rate
        # TPR = TP / (TP + FN)
        # # Specificity or true negative rate
        # TNR = TN / (TN + FP)
        # # Precision or positive predictive value
        # PPV = TP / (TP + FP)
        # # Negative predictive value
        # NPV = TN / (TN + FN)
        # # Fall out or false positive rate
        # FPR = FP / (FP + TN)
        # # False negative rate
        # FNR = FN / (TP + FN)
        # # False discovery rate
        # FDR = FP / (TP + FP)
        return 1.0 * tn / (fp + tn), 1.0 * tp / (fn + tp), 1.0 * (tp + tn) / (tp + fp + fn + tn)
    except Exception as e:
        return np.nan, np.nan, np.nan  # [1][1]; too few values