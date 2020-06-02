import sys

import pandas as pd

sys.path.append('/home/lbozarth/PycharmProjects/fakenews/')
import random
from sklearn.model_selection import KFold

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)

def gen_folds(df, source, n=5):
    kf = KFold(n_splits=n, shuffle=True)
    i = 0
    for train_test_index, validation_index in kf.split(df):
        print(len(train_test_index), len(validation_index))
        traintest, validation = df.iloc[train_test_index], df.iloc[validation_index]
        traintest.reset_index(inplace=True)
        validation.reset_index(inplace=True)

        traintest = traintest[['doc_id', 'domain', 'domain_type']]
        traintest.to_csv("../data/clf_meta/folds_idx/valarch/%s/basic/traintest_fold%s.csv"%(source, i), header=True, index=False)

        validation = validation[['doc_id', 'domain', 'domain_type']]
        validation.to_csv("../data/clf_meta/folds_idx/valarch/%s/basic/validation_fold%s.csv"%(source, i), header=True, index=False)
        i+=1
    return

def gen_folds_domains(df, source, n=10):
    domains = list(df['domain'].unique())
    samp_num = int(len(domains)*0.1)
    for i in range(n):
        samp_domains = random.sample(domains, samp_num)
        print('number of unique domains', len(domains), samp_num)
        print('sampled domains', samp_domains)

        traintest = df[~df['domain'].isin(samp_domains)].reset_index()
        validation = df[df['domain'].isin(samp_domains)].reset_index()

        traintest = traintest[['doc_id', 'domain', 'domain_type']]
        traintest.to_csv("../data/clf_meta/folds_idx/valarch/%s/bydomains/traintest_fold%s.csv"%(source, i), header=True, index=False)

        validation = validation[['doc_id', 'domain', 'domain_type']]
        validation.to_csv("../data/clf_meta/folds_idx/valarch/%s/bydomains/validation_fold%s.csv"%(source, i), header=True, index=False)
    return

def gen_folds_forecast(df, source, rand_dates):
    for rand_date in rand_dates:
        shockd_str = rand_date.strftime("%Y-%m-%d")
        traintest = df[df['min_date'] <= rand_date]
        traintest = traintest[['doc_id', 'domain', 'domain_type', 'min_date']]
        traintest.to_csv("../data/clf_meta/folds_idx/valarch/%s/byforecast/traintest_fold%s.csv"%(source, shockd_str), header=True, index=False)

        validation = df[df['min_date'] > rand_date]
        validation = validation[['doc_id', 'domain', 'domain_type', 'min_date']]
        validation.to_csv("../data/clf_meta/folds_idx/valarch/%s/byforecast/validation_fold%s.csv"%(source, shockd_str), header=True, index=False)
    return

if __name__ == '__main__':
    pass