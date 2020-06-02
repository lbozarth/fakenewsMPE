import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='textacy')

import logging
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from src_classifiers_BTC import MyBasicNLP, MyTextPosTags, TextStatistics
from src_clf_meta import liwc
import multiprocessing as mp
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)

# # Uncertainty	LIWC word category
# # Other reference	LIWC word category
# Objectification	LIWC word category	Power
# Generalizing Terms	LIWC word category
# Self Reference	LIWC word category
# Group Reference	LIWC word category	We
liwceval = liwc.LiwcEvaluator()
def get_liwc_features(text):
    res = liwceval.count_cat(text, ret='dict', divide=False)
    res = {x:y for x,y in res.items() if x in ['posemo', 'negemo', 'power', 'i', 'we', 'sixltr', 'tentav', 'certain', 'focuspast', 'focuspresent', 'focusfuture', 'social', 'cogproc', 'percept', 'affect', 'motion', 'space']}
    # print(res)
    return res

# Spatio-temporal  Infomation	places = GeoText(check_that).country_mentions Perceptual Information
from geotext import GeoText
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# sid = SentimentIntensityAnalyzer()
def get_other_features(text):
    geof =  GeoText(text).country_mentions
    # sentif = sid.polarity_scores(text)
    # sentif.pop('neu', None)
    # # sentif.pop('compound', None)
    # sentif.update(geof)
    return geof

def get_glove_model():
    fpn = "../../shared/data_static/glove.6B/glove.6B.300d.txt"
    embeddings_dict = {}
    with open(fpn, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    # print(embeddings_dict['politician'])
    return embeddings_dict

stop_words = stopwords.words('english')
# model = get_glove_model()
def gen_word2vec_features(text):
    article_tokens = word_tokenize(text)
    filtered_tokens = [w for w in article_tokens if not w in stop_words and w.isalpha()]
    article_vectors = [model[w] for w in filtered_tokens if w in model]
    article_vectors = np.array(article_vectors).sum(axis=0)
    # print(len(article_vectors))
    # print(article_vectors)
    features = {}
    for i in range(len(article_vectors)):
        features["w2v_%s"%i] = article_vectors[i]
    # features = {"w2v_%s"%i:v for i, v in enumerate(article_vectors)}
    # return ",".join(article_vectors)
    # print(features)
    return features

from collections import ChainMap
def gen_features(text):
    try:
        #todo parse domain expertise features
        nltkf = TextStatistics.gen_nltk_features(text)
        posf = MyTextPosTags.gen_pos_features(text)
        nlpf = MyBasicNLP.gen_nlp_features(text)
        liwcf = get_liwc_features(text)
        otherf = get_other_features(text)

        #todo parse word2vec
        # word2vf = gen_word2vec_features(text)

        nltkf.update(posf)
        nltkf.update(nlpf)
        nltkf.update(liwcf)
        # nltkf.update(otherf)
        # nltkf.update(word2vf)
        # features = [nltkf, posf, nlpf, liwcf, otherf, word2vf] #features, word2vf
        # features = dict(ChainMap(*features))
        return nltkf

    except Exception as e:
        print(e)
        return

def get_completed_parts():
    parts = {}
    if not os.path.exists("../data/clf_meta/BTC/parts/"):
        return parts

    for fn in os.listdir("../data/clf_meta/BTC/parts/"):
        pts = fn.split(".cs")[0].split("_")
        source = pts[0]
        if source not in parts:
            parts[source] = []
        doc_id = pts[-1]
        parts[source].append(int(float(doc_id.strip())))
    return parts

completed_parts = get_completed_parts()
print(completed_parts)
def run_single_df_BTC(source, df):
    print('running single df BTC', source)
    if source is None:
        cps = completed_parts['None']
    else:
        cps = completed_parts[source]
    print(cps)
    for doc_id in cps:
        if doc_id in df['doc_id'].tolist():
            print('file already exists', doc_id)
            return

    df['doc'] = df.apply(lambda x: str(x['title']) + " " + str(x['content']), axis=1)
    # print(df['doc_id'].tolist()[:2])
    df['features'] = df['doc'].apply(gen_features)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    if df.empty:
        return

    features = df['features'].tolist()
    features = pd.DataFrame(features)
    df = df[['doc_id']]
    assert(len(df.index)==len(features.index))
    df1 = pd.concat([df, features], axis=1,  sort=False)
    df1.fillna(0, inplace=True)
    # print(df.head(2))
    if not df.empty:
        try:
            wfn = "../data/clf_meta/BTC/parts/%s_part_%s.csv"%(source, df['doc_id'].tolist()[-1])
            # print(wfn)
            df1.to_csv(wfn, header=True, index=False)
        except Exception as e:
            print('error', df.head())
            # print(df.head(2))
            # print(features.head(2))
    return

from functools import partial
#todo preprocessing and remove all news media tokens
def gen_data_all_BTC(source, num_pools=2, chunksize=10, nrows=20):
    if source is not None:
        print('generating from source', source)
        if nrows >0:
            dfs = pd.read_csv("../data/clf_meta/%s_content_all.csv"%source, header=0, sep="\t",
                             usecols=['doc_id', 'title', 'content'], nrows=nrows, iterator=True, chunksize=chunksize)
        else:
            dfs = pd.read_csv("../data/clf_meta/%s_content_all.csv"%source, header=0, sep="\t",
                             usecols=['doc_id', 'title', 'content'], iterator=True, chunksize=chunksize)

        print('num_pools, chunksize, nrows', num_pools, chunksize, nrows)
        p = mp.Pool(num_pools)
        func = partial(run_single_df_BTC, source)
        ndfs = p.map(func, dfs)
        # res = pd.concat(ndfs, axis=0)
        # res.to_csv("../data/clf_meta/BTC/%s_all_clean_text.txt"%source, header=True, index=False, sep="\t")
        p.close()
        p.join()
    else:
        if nrows >0:
            dfs = pd.read_csv('../data/clf_meta/content_unique_all_v2.csv', header=0, sep="\t",
                             usecols=['doc_id', 'title', 'content'], nrows=nrows, iterator=True, chunksize=chunksize)
        else:
            dfs = pd.read_csv('../data/clf_meta/content_unique_all_v2.csv', header=0, sep="\t",
                             usecols=['doc_id', 'title', 'content'], iterator=True, chunksize=chunksize)

        print('num_pools, chunksize, nrows', num_pools, chunksize, nrows)
        p = mp.Pool(num_pools)
        func = partial(run_single_df_BTC, source)
        ndfs = p.map(func, dfs)
        # res = pd.concat(ndfs, axis=0)
        # res.to_csv("../data/clf_meta/BTC/all_clean_text_v2_part2.txt", header=True, index=False, sep="\t")
        p.close()
        p.join()
    return

def gen_data_all():
    # source = None
    # gen_data_all_BTC(source, num_pools=4, chunksize=5000, nrows=-1)

    source = 'fakenewscorpus'
    gen_data_all_BTC(source, num_pools=4, chunksize=5000, nrows=-1)
    source = 'nela'
    gen_data_all_BTC(source, num_pools=4
                     , chunksize=5000, nrows=-1)
    return

def test_gen_data():
    source = 'fakenewscorpus'
    gen_data_all_BTC(source)
    source = 'nela'
    gen_data_all_BTC(source)
    source = None
    gen_data_all_BTC(source)
    return

def test_features():
    text = "Republican presidential candidate Ohio Gov. John Kasich speaks at a campaign event Wednesday, March 23, in Wauwatosa, Wis. | AP Photo When Kasich is matched up against Clinton, 45 percent of registered voters nationwide said they would vote for the Ohio governor, compared to 39 percent for the former secretary of state. Against Clinton, Kasich leads with men, voters between the ages of 18 and 54and white non-Hispanics, while Clinton holds a narrower advantage among women and a wider lead among those who are not white. Even Lindsey Graham, who has endorsed Cruz despite their own colorful history of disagreements, said Thursday that Kasich would be the best presidential candidate and a better president. ""I think John Kasich would be the best nominee, but he doesn't have a chance,"" Graham said on MSNBC's ""Morning Joe,"" adding, ""John Kasich's problem is he is an insider in an outsider year, and nobody seems to want to buy that."" In a theoretical three-way race between Trump, Clinton and Libertarian candidate Gary Johnson, Clinton earned 42 percent, Trump 34 percent and Johnson 11 percent. The Monmouth poll also suggests that Kasich and Bernie Sanders are the only candidates still in the race who are seen more favorably than not. Whereas Clinton had a net negative rating of -11 points (40 percent positive to 51 percent negative) and Trump is far below at -30 points (30 percent to 60 percent), Kasich's favorability sits at +32 points (50 percent to 18 percent), though 32 percent said they had no opinion of him. Approval of Cruz, while still a net negative 6 points (37 percent to 43 percent), has risen 8 points since October and 12 points since last June. Monmouth conducted its poll March 17-20 via landlines and cellphones, surveying 1,008 adults nationwide, including 848 registered voters. For that subsample, the margin of error is plus or minus 3.4 percentage points."
    # get_other_features(text)
    features = gen_features(text)
    print(len(features), features)
    print(features.keys())

def gen_data_all_BTC_text_parts():
    srcs = [None, 'fakenewscorpus', 'nela']
    for source in srcs:
        resss = []
        print('gen data from', source)
        for fn in os.listdir("../data/clf_meta/BTC/parts"):
            if str(source) not in fn:
                continue
            fpn = os.path.join("../data/clf_meta/BTC/parts", fn)
            print(fpn)
            df = pd.read_csv(fpn, header=0)
            resss.append(df)

        resss = pd.concat(resss, axis=0)
        if source:
            resss.to_csv("../data/clf_meta/BTC/%s_all_features.txt"%source, header=True, index=False, sep="\t")
        else:
            resss.to_csv("../data/clf_meta/BTC/all_features.txt", header=True, index=False, sep="\t")

if __name__ == '__main__':
    test_features()
    # test_gen_data()
    # gen_data_all()
    # gen_data_all_BTC_text_parts()
    pass