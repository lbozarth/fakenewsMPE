# https://github.com/xuexue/DataInColour/blob/master/utils/liwc.py

import nltk
from collections import defaultdict


class Dictionary:
    def __init__(self, letter):
        self.letter = letter
        self.children = {}
        self.cats = []

    def addcats(self, cats):
        self.cats = cats

    def addchild(self, word, cats):
        if len(word) == 0:
            self.addcats(cats)
        else:
            child = self.children.get(word[0])
            if child is None:
                child = Dictionary(word[0])
                self.children[word[0]] = child
            child.addchild(word[1:], cats)

    def getchild(self, l):
        child = self.children.get(l)
        if child is None:
            child = self.children.get('*')
        return child

    def get(self, word):
        if self.letter == '*':
            return self.cats
        if word == '':
            if len(self.cats) > 0:
                return self.cats
            star = self.getchild('*')
            if star is not None:
                return star.cats
            return None
        child = self.getchild(word[0])
        if child is not None:
            return child.get(word[1:])
        return None


class LiwcEvaluator:
    categories = {}
    dictionary = Dictionary('')
    _CATFILE = '../data/lexicons/LIWC2015_cat.txt'
    _DICTFILE = '../data/lexicons/LIWC2015.dic'
    tknzr = nltk.tokenize.TweetTokenizer()

    def __init__(self):
        for r in open(self._CATFILE): #, encoding="ISO-8859-1"
            if r.find('%') < 0:
                id, name, full_name = r.rstrip('\n\t\r ').split('\t')
                self.categories[id] = name.split('@')[0]
        #encoding="ISO-8859-1"
        for line in open(self._DICTFILE): #, encoding="ISO-8859-1"
            w = line.rstrip('\n\t\r').split('\t')
            self.dictionary.addchild(w[0], w[1:])

    def header(self):
        return list(self.categories.values()) + ['n']

    def divide_words(self, counts):
        nwords = counts['n']
        for cat in self.header():
            if cat != 'n':
                counts[cat] = float(counts[cat]) / nwords

    def count_cat(self, words, ret="list", divide=False):
        # tokenize

        words = self.tknzr.tokenize(words.lower())
        # iterate through allWords words
        counts = defaultdict(lambda: 0)
        for word in words:
            if word.isalpha():
                counts['n'] += 1

                cats = self.dictionary.get(word)
                if cats is not None:
                    for cat in cats:
                        try:
                            counts[self.categories[cat]] += 1
                        except:
                            pass

        if divide:
            if counts['n'] == 0:
                counts['n'] = 1
            self.divide_words(counts)

        if ret == "dict":
            return counts
        return map(lambda n: counts[n], self.header())
