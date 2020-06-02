import sqlite3
import logging
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def transform_glove_to_word2vec(glove_file, output_file):
    glove2word2vec(glove_input_file=glove_file, word2vec_output_file=output_file)


def pickle_loader(pkl_file):
    """Reads all lines of a pickle file
    Args:
        pkl_file: A pickle file to be traversed
    """
    try:
        while True:
            yield pickle.load(pkl_file)
    except EOFError:
        pass


def load_pickle_all(pkl_file):
    with open(pkl_file) as f:
        vectors = []
        for pickle_vecs in pickle_loader(f):
            vectors.append(pickle_vecs)

    return np.array(vectors)


def split_features_from_targets(pkl_file, features_out, targets_out):
    """Splits the features from the targets in a pickle file and saves them as npy files.
    Args:
        pkl_file: A pickle file that has both targets and features saved in an npy array.
        features_out: the path to save the features npy file
        targets_out: the path to save the targets npy file
    """
    with open(pkl_file) as f:
        X = []
        y = []
        for pickle_vecs in pickle_loader(f):
            X.append(np.array(pickle_vecs[:-1]))
            y.append(np.array(pickle_vecs[-1]))
        X = np.array(X)
        y = np.array(y)
        np.save(features_out, X)
        np.save(targets_out, y)


def split_features_targets_ids(pkl_file):
    """Splits the features from the targets in a pickle file and returns X, y, ids.
    Args:
        pkl_file: A pickle file that has features targets and ids.
    """
    with open(pkl_file) as f:
        X = []
        y = []
        ids = []
        for pickle_vecs in pickle_loader(f):
            X.append(pickle_vecs[:-2])
            y.append(pickle_vecs[-2])
            ids.append(pickle_vecs[-1])
        return X, y, ids


def preprocess_articles_to_df(vectors_path, bin, data_path, output_file):
    """Transform articles from a db to vectors with a given vector model e.g word2vec or GloVe and sum the vectors
    of each article saving them as pickle file.
    Args:
        vectors_path: The path to a vectors_model e.g a word2vec pretrained model.
        bin: True, False depending on the whether or not the vectors are in binary.
        db: the database to get the articles from
        output_file: The pickle file to save the calculated vectors to
    """

    # load model
    model = KeyedVectors.load_word2vec_format(vectors_path, binary=bin)
    stop_words = set(stopwords.words('english'))

    data = pd.read_csv(data_path)

    cursor = pd.DataFrame(data)

    check_encoding_errors = 0
    # final_dict = {}
    temp_dict = {}
    for index, item in cursor.iterrows():

        text = item[1]
        if isinstance(text, str):
            try:
                text = unicode(text, "utf-8")
            except UnicodeError as e:
                print e
                check_encoding_errors = check_encoding_errors + 1
                continue
        article_tokens = word_tokenize(text.lower())
        if len(article_tokens) > 1:
            filtered_tokens = [w for w in article_tokens if not w in stop_words and w.isalpha()]
            article_vectors = [model[w] for w in filtered_tokens if w in model]
            article_vectors = np.array(article_vectors).sum(axis=0)
            article_vectors = np.append(article_vectors, [item[0], item[2]])
            temp_dict[index] = article_vectors
            # final_article_vectors = np.append(article_vectors, (item[0], item[2]))
            # pickle.dump(final_article_vectors, output)

    df = pd.DataFrame.from_dict(data=temp_dict, orient="index")
    df.to_csv("new_test.csv")


def preprocess_articles_db(vectors_path, bin, db, output_file):
    """Transform articles from a db to vectors with a given vector model e.g word2vec or GloVe and sum the vectors
    of each article saving them as pickle file.
    Args:
        vectors_path: The path to a vectors_model e.g a word2vec pretrained model.
        bin: True, False depending on the whether or not the vectors are in binary.
        db: the database to get the articles from
        output_file: The pickle file to save the calculated vectors to
    """

    # load model
    model = KeyedVectors.load_word2vec_format(vectors_path, binary=bin)
    stop_words = set(stopwords.words('english'))

    # connect to db
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    with open(output_file, 'wb') as output:
        for article in cursor.execute('SELECT text, rating, id  FROM RawData').fetchall():
            article_tokens = word_tokenize(article[0].lower())
            if len(article_tokens) > 1:
                filtered_tokens = [w for w in article_tokens if not w in stop_words and w.isalpha()]
                article_vectors = [model[w] for w in filtered_tokens if w in model]
                article_vectors = np.array(article_vectors).sum(axis=0)
                final_article_vectors = np.append(article_vectors, [article[1], article[2]])
                pickle.dump(final_article_vectors, output)

    connection.close()