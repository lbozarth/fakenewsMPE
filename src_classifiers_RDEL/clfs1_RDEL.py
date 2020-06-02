import pandas as pd
import numpy as np
import sys,os,socket
hostname = socket.gethostname()
if 'arc-ts.umich.edu' in hostname:
    sys.path.append('/scratch/cbudak_root/cbudak/lbozarth/PycharmProjects/fakenews/')
else:
    sys.path.append('/home/lbozarth/PycharmProjects/fakenews/')

from src_clf_meta.clfs_common import *

import multiprocessing as mp
from functools import partial
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

CLF_NAME = 'RDEL'
DEFAULT_CLF_TESTING = False
DEFAULT_INCRE_RANGES = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)

#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
####TODO Feature Generation Functions
primary_events = [('2016-05-03', '05/03', 'nomination', 'trump'), ('2016-05-26', '05/26', 'nomination', 'trump'),
                  ('2016-06-06', '06/06', 'nomination', 'clinton')]  # cruz dropout, trump has enough votes, clinton has enough votes
ge_events = [("2016-07-18", '07/18', "convention", 'trump'), ("2016-07-28", '07/28', "convention", 'clinton'),  # RNC, DNC
             ("2016-09-09", '09/09', "scandal", 'clinton'),  # deplorables, sick
             ("2016-10-07", '10/07', "scandal", 'both'),  # trump tape, clinton email, 2nd debate
             ("2016-09-26", '09/26', "debate", 'both'), ("2016-10-09", '10/09', "debate", 'both'), ("2016-10-19", '10/19', "debate", 'both'),
             ("2016-10-28", '10/28', 'scandal', 'clinton'), ("2016-11-06", '11/06', 'scandal', 'clinton'),
             ("2016-11-08", '11/08', 'election', 'both')
             ]  # email, email/election
events = {'primary': primary_events, 'general election': ge_events}
def get_shock_type2(x):
    if x in ['convention', 'debate', 'election', 'nomination']:
        return 'scheduled'
    if x in ['scandal']:
        return 'scandal'
    print('unexpected')
    return

from datetime import datetime
def gen_event_shocks():
    subsets = []
    for period, shocks in events.items():
        # "2016-10-28", '10/28', 'scandal', 'clinton'
        for ddate, day, tp, cand in shocks:
            subsets.append([period, day, ddate, tp, cand])
    subsets = pd.DataFrame(subsets, columns=['period', 'shock_label', 'shock_date', 'shock_type', 'candidate'])
    subsets['shock_date'] = subsets['shock_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    subsets['shock_type2'] = subsets['shock_type'].apply(get_shock_type2)
    subsets2 = subsets.groupby(['shock_type2'], as_index=False).agg({'shock_date':list})
    subsets2.loc[2] = ['noevent', [datetime.strptime('2016-08-10', "%Y-%m-%d"), datetime.strptime('2016-09-02', "%Y-%m-%d")]]
    # print(subsets2)
    return subsets2


def gen_folds(source='mediabiasfactcheck', n=3):
    for idx in range(n):
        fpn = "../data/clf_meta/folds_idx/source_%s_traintest_fold%s.csv" % (source, idx)
        df = pd.read_csv(fpn, header=0)
        print('len0', len(df.index))

        ksplit = StratifiedShuffleSplit(n_splits=n, test_size=0.2)
        for train_index, test_index in ksplit.split(df, df['domain_type']):
            # print(train_index[:5], test_index[:5])
            print(len(train_index), len(test_index))

            X_train, X_test = df.iloc[train_index], df.iloc[test_index]
            X_train.reset_index(inplace=True)
            X_test.reset_index(inplace=True)

            # X_train.to_csv("../data/clf_meta/RDEL/folds_idx/source_%s_train_fold%s.csv"%(source, idx), header=True, index=False)
            # X_test.to_csv("../data/clf_meta/RDEL/folds_idx/source_%s_test_fold%s.csv" % (source, idx), header=True, index=False)
            print('len1', len(X_train.index), len(X_test.index))
    return

def basic_vec(key_col, docs, max_df=0.25, min_df=100, max_features=2000, use_idf=True, ngram_range=(1,2)):
    # docs = [clean_text(x) for x in docs]
    cnt = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features,
                          use_idf=use_idf) #, stop_words=all_stop_words, tokenizer=my_tokenizer

    X = cnt.fit_transform(docs)
    # print(cnt.get_feature_names())
    count_vect_df = pd.DataFrame(X.todense(), columns=cnt.get_feature_names())
    # print(count_vect_df.head(2))
    print('len_vec', len(key_col.index), len(count_vect_df.index))
    assert(len(key_col.index) == len(count_vect_df.index))
    key_col = pd.concat([key_col, count_vect_df], axis=1)
    # print(key_col.head(2))
    return cnt, key_col

def basic_vec_fit(key_col, docs, cnt):
    X = cnt.transform(docs)
    count_vect_df = pd.DataFrame(X.todense(), columns=cnt.get_feature_names())
    assert(len(key_col.index) == len(count_vect_df.index))
    key_col = pd.concat([key_col, count_vect_df], axis=1)
    return cnt, key_col

def gen_cosine_smilarity_all(titles, content):
    titles = titles.values
    contents = content.values
    resss = []
    for i in range(len(titles)):
        # print(titles[i].reshape(1, -1), contents[i].reshape(1, -1))
        cs = cosine_similarity(titles[i].reshape(1, -1), contents[i].reshape(1, -1))
        # print(cs)
        resss.append(cs[0])
        # sys.exit()

    resss = pd.DataFrame(resss, columns=['cosine_similarity'])
    # print(resss.head())
    return resss

def genXy_common(fpn, df, label_balance=True, label_ids=['domain_type']):
    if 'validation' not in fpn: # this is training set
        if label_balance:
            domain_count_min = np.min(df.groupby(label_ids)['doc_id'].count().tolist())
            print('min doc count is', domain_count_min)
            df = df.groupby(label_ids).apply(lambda x: x.sample(n=domain_count_min))
            print('len2', len(df.index))
            df.reset_index(inplace=True, drop=True)
    return df

def get_docs_df(contents, fpn, use_hard_min=False, label_balance=True):
    df = pd.read_csv(fpn, header=0, usecols=['doc_id', 'domain', 'domain_type'])
    print('len0', len(df.index))
    df = pd.merge(df, contents, on='doc_id')
    # print(df.head())
    print('len1', len(df.index))

    df = genXy_common(fpn, df, label_balance=label_balance)
    # if 'validation' not in fpn:
    #     domain_count_min = np.min(df.groupby('domain_type')['doc_id'].count().tolist())
    #     if use_hard_min:
    #         domain_count_min = min(domain_count_min, 50000)
    #     print('min doc count is', domain_count_min)
    #     df = df.groupby('domain_type').apply(lambda x: x.sample(n=domain_count_min))
    #     print('len2', len(df.index))
    #     df.reset_index(inplace=True, drop=True)
    # else:
    #     # print('use_hard_min', use_hard_min)
    #     if len(df.index)>100000 and use_hard_min:
    #         df = df.groupby('domain_type').apply(lambda x: x.sample(frac=0.5))
    #         print('len2', len(df.index))
    #         df.reset_index(inplace=True, drop=True)

    print('doc shapes', df.shape)
    df = pd.melt(df, id_vars=['doc_id', 'domain', 'domain_type'], var_name='doc_type', value_name='doc')
    # print(df.head())
    # print(df.groupby('doc_type').count())
    # df.sort_values('doc_id', inplace=True)
    # print(df.head())
    return df

DEFAULT_MAX_FEATURES = 2000
def gen_features_test(dfc, wfn, vecfn, cnt=None, istest=True):
    # print('len before', len(dfc.index))
    dfc['doc'].fillna("", inplace=True)
    # # dfc = dfc[~dfc['doc'].isnull()].reset_index()
    # # dfc = dfc[dfc['doc'].str.count(" ")>=2].reset_index()
    # print('len after', len(dfc.index))
    docs = dfc['doc'].tolist()

    if cnt is not None:
        cnt, key_col = basic_vec_fit(dfc[['doc_id', 'doc_type', 'domain_type']], docs, cnt)
    elif (vecfn is not None) and istest:
        cnt = pickle.load(open(vecfn, 'rb'))
        cnt, key_col = basic_vec_fit(dfc[['doc_id', 'doc_type', 'domain_type']], docs, cnt)
    else:
        cnt, key_col = basic_vec(dfc[['doc_id', 'doc_type', 'domain_type']], docs, max_features=DEFAULT_MAX_FEATURES)
        print('shape', key_col.shape)
        if cnt:
            pickle.dump(cnt, open(vecfn, 'wb'))

    actual_max_features = min(DEFAULT_MAX_FEATURES, key_col.shape[1]-3)
    print('actual number of features', actual_max_features)
    xcols = list(key_col.columns)[-actual_max_features:]
    # print(xcols)
    titles = key_col[key_col['doc_type'] == 'title_clean'].reset_index(drop=True)
    titles = titles[xcols]
    titles.columns = [x + "_title" for x in titles.columns]
    content = key_col[key_col['doc_type'] == 'content_clean'].reset_index(drop=True)
    content = content[xcols]
    content.columns = [x + "_content" for x in content.columns]
    print('title, content shapes', titles.shape, content.shape)
    # print(titles.head(2))
    # print(content.head(2))
    cosine = gen_cosine_smilarity_all(titles, content)

    print(key_col.columns[:5])
    key_col = key_col[['doc_id', 'domain_type']]
    # key_col.sort_values('doc_id', inplace=True)
    key_col.drop_duplicates(inplace=True)
    key_col.reset_index(drop=True, inplace=True)
    # print(key_col.head())
    print(len(key_col.index), len(titles.index), len(content.index), len(cosine.index))
    assert (len(key_col.index) == len(titles.index) == len(content.index) == len(cosine.index))
    dff = pd.concat([key_col, titles, content, cosine], axis=1)
    dff.to_csv(wfn, header=True, index=False)
    return cnt

def gen_feature_train(dfc, wfn, vecfn):
    return gen_features_test(dfc, wfn, vecfn, None, False)


def get_features(dataset):
    if dataset == 'default':
        features = pd.read_csv('../data/clf_meta/RDEL/all_clean_text_v2.txt', header=0, sep="\t")
    else:
        features = pd.read_csv('../data/clf_meta/RDEL/%s_all_clean_text.txt'%dataset, header=0, sep="\t")
    return features

def run_fold_features(contents, vecfn, trainfpn, testfpn, trainwfpn, testwfpn, dataset,
                      use_hard_min=False, run_validation=True, label_balance=True):
    if contents is None:
        contents = get_features(dataset)

    print(dataset, trainfpn, testfpn, label_balance)
    if not os.path.exists(vecfn) or not os.path.exists(trainwfpn):
        print('generate vectorizer and train features')
        df = get_docs_df(contents, trainfpn, use_hard_min=use_hard_min, label_balance=label_balance)
        cnt = gen_feature_train(df, trainwfpn, vecfn)
    else:
        print('loading vectorizer')
        cnt = pickle.load(open(vecfn, 'rb'))

    if not os.path.exists(testwfpn):
        print('generating test features')
        df = get_docs_df(contents, testfpn, use_hard_min=use_hard_min, label_balance=label_balance)
        gen_features_test(df, testwfpn, None, cnt)
    return


#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
####TODO Classifier Function
def gen_lr(X, y):
    param_grid = {'C': [0.1, 1], 'penalty': ['l2'], 'solver': ['sag']}

    use_n_jobs = 1
    hostname = socket.gethostname()
    print('hostname is', hostname)
    if 'arc-ts.umich.edu' in hostname:
        use_n_jobs = 1
    print('using number of jobs', use_n_jobs)

    clf = GridSearchCV(LogisticRegression(max_iter=1000, tol=0.001, class_weight='balanced'), scoring='roc_auc',
                       param_grid=param_grid, cv=StratifiedKFold(), n_jobs=use_n_jobs)

    clf.fit(X, y)
    print('\n Training completed', clf.best_score_)
    return clf.best_estimator_

def gen_mlp(X, y):
    parameter_space = {
        'hidden_layer_sizes': [(100, 1000)],  # , (500,), (1000,), (1500,)],
        'alpha': [0.0001, 0.05],
    }

    use_n_jobs = 1
    hostname = socket.gethostname()
    print('hostname is', hostname)
    if 'arc-ts.umich.edu' in hostname:
        use_n_jobs = 1
    print('using number of jobs', use_n_jobs)

    X.fillna(0, inplace=True)
    clf = GridSearchCV(MLPClassifier(), param_grid=parameter_space, cv=StratifiedKFold(), scoring='roc_auc', n_jobs=use_n_jobs)
    clf.fit(X, y)

    # parameters = {'alpha': [0.0001], 'hidden_layer_sizes': [200], 'learning_rate_init': [0.005],
    #               'early_stopping': [True]}
    # clf = GridSearchCV(MLPClassifier(), param_grid=parameters, cv=StratifiedKFold())
    # clf = MLPClassifier(learning_rate_init=0.005, hidden_layer_sizes=(2000,), alpha=0.0001, verbose=True,
    #                     early_stopping=True)
    print('\n Training completed best score', clf.best_score_)
    return clf.best_estimator_

def gen_dt(X, y):
    rfc = RandomForestClassifier(max_features='sqrt', n_estimators=50, oob_score=True)
    param_grid = {'n_estimators': [100, 500], 'max_features': ['auto', 'sqrt', 'log2']}
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=StratifiedKFold(), scoring='roc_auc')
    CV_rfc.fit(X, y)
    print('\n Training completed', CV_rfc.best_score_)
    return CV_rfc.best_estimator_

unwantedlst = []
for ul in ['sputnik', 'uk','reuter','ap','amazon','mr','continu read', 'main stori', 'read main', 'wa post', 'infowar', 'reuter', 'sun', 'cb', 'www uk', "www", 'der', 've', 'subscrib sign', 'subscrib', 'newslett subscrib', 'subscrib error', 'newslett subscrib', 'thank subscrib', 'robot', 'fox', 'getti', 'rt', 'la']:
    unwantedlst.append(ul+"_content")
    unwantedlst.append(ul + "_title")
def getXy(fpn, use_hard_min=False, label_balance=True):
    df = pd.read_csv(fpn, header=0)
    df['domain_type'] = df['domain_type'].apply(lambda x: 1 if x == 'fake' else 0)
    print('len1', len(df.index))

    if ('validation' not in fpn) and (label_balance==False) and (len(df.index)>=100000):
        df = df.groupby('domain_type').apply(lambda x: x.sample(frac=0.25))
        print('len3', len(df.index))
        df.reset_index(inplace=True, drop=True)

    print(df.shape)
    # print(df.columns)
    neg_index = DEFAULT_MAX_FEATURES * 2 + 1

    X = df[df.columns[-neg_index:]]
    y = df['domain_type']
    # print('y unique', y.unique())
    # xcols = [col for col in X.columns if col not in unwantedlst]
    # print('number of features', len(xcols))
    # X = X[xcols]
    cunwanted = []
    for unwanted in unwantedlst:
        if unwanted in X.columns:
            cunwanted.append(unwanted)

    X.drop(cunwanted, axis=1, inplace=True)

    print('x shape', X.shape)
    return df['doc_id'],X,y

def run_fold(clfwfn, trainfn, testfn, wfn, dataset, idx_type, idx, clf_type, clf_fun, use_hard_min=False, run_validation=False, label_balance=True):
    if clf_fun is None:
        clf_type = 'mlp'
        clf_fun = gen_mlp

    print('run_fold', dataset, clfwfn, trainfn, testfn, label_balance)
    # todo check clf performance on validation test
    if os.path.exists(clfwfn) and os.path.exists(trainfn):
        print('loading clf for file', clf_type)
        clf = pickle.load(open(clfwfn, 'rb'))
    else:
        print('generate clf', clf_type)
        print(trainfn)
        _, X, y = getXy(trainfn, use_hard_min=use_hard_min)
        clf = clf_fun(X, y)
        pickle.dump(clf, open(clfwfn, 'wb'))

    if not run_validation:
        return

    print('testing on validation', testfn)
    doc_ids, test_X, test_y = getXy(testfn, use_hard_min=use_hard_min)

    y_pred = clf.predict(test_X)
    y_pred_prob = clf.predict_proba(test_X)
    y_pred_prob = [p[1] for p in y_pred_prob]
    resss = list(zip(doc_ids, test_y, y_pred, y_pred_prob))
    resss = pd.DataFrame(resss, columns=['doc_id', 'y_true', 'y_pred', 'y_pred_prob'])
    # print(resss.head())
    resss['clf'] = 'RDEL'
    resss['clf_type'] = clf_type
    resss['fold'] = idx
    resss['fold_type'] = idx_type
    resss['dataset'] = dataset
    resss.to_csv(wfn, header=True, index=False)

    performance = evaluate_clf_preformance(test_y, y_pred, y_pred_prob)
    performance['clf'] = 'RDEL'
    performance['fold_type'] = idx_type
    performance['fold'] = idx
    performance['clf_type'] = clf_type
    performance['dataset'] = dataset
    return performance

def run_clfs(dataset='default', source='mediabiasfactcheck', n=3, use_hard_min=False, is_incremental=False, run_validation=False, range_start=1):
    if dataset == 'default':
        fn_part = 'source_%s_'%source
    else:
        fn_part = ""

    if is_incremental:
        print('getting incremental')
        perfs = []
        for ptg in DEFAULT_INCRE_RANGES:
            if ptg < range_start:
                continue
            for idx in range(n):
                idx_type = 'ptg%s' % ptg
                print('running', idx_type)
                for clf_type, clf_fun in {'mlp':gen_mlp}.items(): #
                    clfwfn = "../data/clf_meta/RDEL/models/increment/%s/%s_%straintest_ptg%s_fold%s.sav" % (
                    dataset, clf_type, fn_part, ptg, idx)
                    trainfpn = "../data/clf_meta/RDEL/features/increment/%s/%straintest_ptg%s_fold%s.csv" % (dataset, fn_part, ptg, idx)
                    testfpn = "../data/clf_meta/RDEL/features/increment/%s/%svalidation_ptg%s_fold%s.csv" % (dataset, fn_part, ptg, idx)
                    wfn = "../data/clf_meta/RDEL/preds/increment/%s/%s_%svalidation_ptg%s_fold%s.csv" % (
                    dataset, clf_type, fn_part, ptg, idx)
                    print(clfwfn, trainfpn, testfpn)
                    performance = run_fold(clfwfn, trainfpn, testfpn, wfn, dataset, idx_type, idx, clf_type, clf_fun, use_hard_min, run_validation)
                    if performance:
                        perfs.append(performance)
                    if DEFAULT_CLF_TESTING:
                        return

            #         break
            #     break
            # break
        if perfs and run_validation:
            perfs = pd.DataFrame(perfs)
            wfn = "../data/clf_meta/RDEL/performance_results_%s_increment.csv"%(dataset)
            perfs.to_csv(wfn, header=True, index=False)

    else:
        perfs = []
        for idx in range(n):
            for clf_type, clf_fun in {'mlp':gen_mlp}.items(): #
                clfwfn = "../data/clf_meta/RDEL/models/%s/%s_%straintest_fold%s.sav" % (dataset, clf_type, fn_part, idx)
                trainfpn = "../data/clf_meta/RDEL/features/%s/%straintest_fold%s.csv" % (dataset, fn_part, idx)
                testfpn = "../data/clf_meta/RDEL/features/%s/%svalidation_fold%s.csv" % (dataset, fn_part, idx)
                wfn = "../data/clf_meta/RDEL/preds/%s/%s_%svalidation_fold%s.csv" % (dataset, clf_type, fn_part, idx)
                print(clfwfn, trainfpn, testfpn)
                performance = run_fold(clfwfn, trainfpn, testfpn, wfn, dataset, '80-20', idx, clf_type, clf_fun, run_validation)
                if performance:
                    perfs.append(performance)
                if DEFAULT_CLF_TESTING:
                    return

        if perfs and run_validation:
            perfs = pd.DataFrame(perfs)
            wfn = "../data/clf_meta/RDEL/performance_results_%s.csv"%dataset
            perfs.to_csv(wfn, header=True, index=False)
    return

def run_clfs_shocks_default(source='mediabiasfactcheck', run_validation=False):#
    perfs = []
    #todo this is by dates
    shocks_date = gen_event_shocks()
    # print(shocks_date.values)
    for shock_type,shock_days in shocks_date.values:
        for shockd in shock_days:
            shockd_str = shockd.strftime("%Y-%m-%d")
            for clf_type, clf_fun in {'mlp': gen_mlp}.items():  #'lr': gen_lr,
                clfwfn = "../data/clf_meta/RDEL/models/events_v2/%s_source_%s_traintest_%s_%s_%s.sav" % (clf_type,
                        source, shock_type, shockd_str, 0)
                if os.path.exists(clfwfn):
                    print('loading clf for file', clf_type)
                    clf = pickle.load(open(clfwfn, 'rb'))
                else:
                    print('generate clf', clf_type, clfwfn)
                    fpn = "../data/clf_meta/RDEL/features/events_v2/source_%s_traintest_%s_%s_%s.csv" % (
                source, shock_type, shockd_str, 0)
                    _, X, y = getXy(fpn)
                    clf = clf_fun(X, y)
                    pickle.dump(clf, open(clfwfn, 'wb'))

                if not run_validation:
                    continue

                # todo check clf performance on validation test
                for limdays in [5, 7, 3]:
                    fpn = "../data/clf_meta/RDEL/features/events_v2/source_%s_validation_%s_%s_%s.csv" % (
                source, shock_type, shockd_str, limdays)
                    print(fpn)
                    doc_ids, test_X, test_y = getXy(fpn)

                    y_pred = clf.predict(test_X)
                    y_pred_prob = clf.predict_proba(test_X)
                    y_pred_prob = [p[1] for p in y_pred_prob]
                    performance = evaluate_clf_preformance(test_y, y_pred, y_pred_prob)
                    performance['clf'] = 'RDEL'
                    performance['clf_type'] = clf_type
                    performance['shock_type'] = shock_type
                    performance['shockd'] = shockd_str
                    performance['limday'] = limdays
                    print(performance)
                    perfs.append(performance)

                    if DEFAULT_CLF_TESTING:
                        return

    perfs = pd.DataFrame(perfs)
    wfn = "../data/clf_meta/RDEL/performance_results_shocks_v2.csv"
    perfs.to_csv(wfn, header=True, index=False)
    return

def run_clfs_forecast_default(source='mediabiasfactcheck', run_validation=False):
    perfs = []
    shock_days = gen_rand_dates()
    for shockd in shock_days:
        shockd_str = shockd.strftime("%Y-%m-%d")
        for clf_type, clf_fun in {'mlp': gen_mlp}.items():  #'lr': gen_lr,
            clfwfn = "../data/clf_meta/RDEL/models/forecast/%s_source_%s_traintest_%s_%s.sav" % (clf_type,
                    source, shockd_str, 0)
            if os.path.exists(clfwfn):
                print('loading clf for file', clf_type)
                clf = pickle.load(open(clfwfn, 'rb'))
            else:
                print('generate clf', clf_type, clfwfn)
                fpn = "../data/clf_meta/RDEL/features/forecast/source_%s_traintest_%s_%s.csv" % (
            source, shockd_str, 0)
                _, X, y = getXy(fpn)
                clf = clf_fun(X, y)
                pickle.dump(clf, open(clfwfn, 'wb'))

            if not run_validation:
                continue

            # todo check clf performance on validation test
            for limdays in [5, 7, 3]:
                fpn = "../data/clf_meta/RDEL/features/forecast/source_%s_validation_%s_%s.csv" % (
            source, shockd_str, limdays)
                print(fpn)
                doc_ids, test_X, test_y = getXy(fpn)

                y_pred = clf.predict(test_X)
                y_pred_prob = clf.predict_proba(test_X)
                y_pred_prob = [p[1] for p in y_pred_prob]
                performance = evaluate_clf_preformance(test_y, y_pred, y_pred_prob)
                performance['clf'] = 'RDEL'
                performance['clf_type'] = clf_type
                performance['shockd'] = shockd_str
                performance['limday'] = limdays
                print(performance)
                perfs.append(performance)

                if DEFAULT_CLF_TESTING:
                    return

    perfs = pd.DataFrame(perfs)
    wfn = "../data/clf_meta/RDEL/performance_results_forecast.csv"
    perfs.to_csv(wfn, header=True, index=False)
    return

def run_clfs_shocks(run_validation=False, use_hard_min=False):
    mkdir_tree('RDEL')
    # gen_features_loop_shock(use_hard_min=use_hard_min)
    run_clfs_shocks_default(run_validation=run_validation)
    return

def run_clfs_forecast(run_validation=False, use_hard_min=False):
    mkdir_tree('RDEL')
    # gen_features_loop_forecast(use_hard_min=use_hard_min)
    run_clfs_forecast_default(run_validation=run_validation)
    return

if __name__ == '__main__':
    pass