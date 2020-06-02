#!/usr/bin/env python
# coding: utf8
"""
Definition: This file contains basic text preprocessing tasks:
1. tokenization:    uses NLTK function for exporting tokens. Input text.Returns a list
2. textNormalization:   sets tokens to lowercase. InputReturns a list
3. stemming:    uses NLTK function for stemming. Returns a list
4. stopwordsRemoveal:   uses NLTK stopword list. Returns a list
5. tokensToDict:    takes as input a list of tokens and returns a dict with the
                    number of occurances


Prerequisites: NLTK
author : ggravan
Created @ 19/12/2017

The paper is called behind the cues
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from textacy import preprocess
from nltk.corpus import reuters
from nltk.probability import FreqDist
import collections
from operator import itemgetter
import heapq

stemmer = SnowballStemmer("english", ignore_stopwords=True)

word_list = reuters.words()
reuters_tokens = [word.lower() for word in word_list if word.isalpha()]
reuters_distribution = FreqDist(word for word in reuters_tokens)


class MyBasicNLP:
    """

    """

    def __init__(self, text):
        """
        @ initialization spacy's fix_bad_unicode method
        is employed for avoiding exceptions in further
        pre-processing
        :param text: A string with the text for pre-processing
        """
        # self.text = preprocess.fix_bad_unicode(text=text, normalization='NFC')
        self.text = text

    def get_token_list(self):
        """

        :return:
        """
        tokens = nltk.word_tokenize(text=self.text, language='english')
        return tokens

    def get_normalized_tokens(self):
        """

        :return:
        """
        normalized_tokens = [word.lower() for word in self.get_token_list() if word.isalpha()]
        return normalized_tokens

    def remove_stopwords(self):
        """

        :return:
        """
        tokens_filtered = [word for word in self.get_normalized_tokens() if word not in stopwords.words('english')]
        return tokens_filtered

    def get_stem_tokens_normalized(self):
        """

        :return:
        """
        tokens = self.get_normalized_tokens()
        stemmed = [stemmer.stem(word) for word in tokens]
        return stemmed

    def get_normalized_tokens_as_dict(self):
        """

        :return:
        """
        tokens = self.get_normalized_tokens()
        word_frequency = [tokens.count(token) for token in tokens]
        return dict(zip(tokens, word_frequency))

    def get_stemmed_tokens_as_dict(self):
        """

        :return:
        """
        stemmed_tokens = self.get_stem_tokens_normalized()
        word_frequency = [stemmed_tokens.count(token) for token in stemmed_tokens]
        return dict(zip(stemmed_tokens, word_frequency))

    def get_filtered_as_dict(self):
        """

        :return:
        """
        tokens = self.remove_stopwords()
        word_frequency = [tokens.count(token) for token in tokens]
        return dict(zip(tokens, word_frequency))

    def get_filtered_words_count(self):
        """

        :return:
        """
        dictionary = self.get_filtered_as_dict()
        return sum(dictionary.values())

    def stop_words_percent(self):
        """

        :return:
        """
        words_before_removal = len(self.get_normalized_tokens())
        words_after_removal = self.get_filtered_words_count()
        try:
            return float(words_before_removal - words_after_removal) / float(words_before_removal)
        except ZeroDivisionError as e:
            # print(e)
            return 0

    def get_sentences_tokenized(self):
        tokens = nltk.sent_tokenize(text=self.text, language='english')
        return tokens

    def get_short_sentences(self):
        short_count = 0
        sentences = self.get_sentences_tokenized()
        for sentence in sentences:
            tokens = nltk.word_tokenize(text=sentence, language='english')
            if len(tokens) <= 10:
                short_count = short_count + 1
        return short_count

    def get_long_sentences(self):
        long_count = 0
        sentences = self.get_sentences_tokenized()
        for sentence in sentences:
            tokens = nltk.word_tokenize(text=sentence, language='english')
            if len(tokens) > 10:
                long_count = long_count + 1
        return long_count

    def get_errors(self):
        count = 0
        word_set = set(reuters_tokens)
        tokens = set(self.get_normalized_tokens())
        for token in tokens:
            if token not in word_set:
                count = count + 1

        return count


    def get_sentence_depth(self):
        sentences = self.get_sentences_tokenized()
        for sentence in sentences:
            nltk.tree


    def get_flu_coca_c(self):

        count = 0
        tokens_freq = self.get_normalized_tokens_as_dict()
        test = self.least_common_values(tokens_freq, 3)
        for k, v in test:
            count = count + reuters_distribution[k]

        return count / 3.0

    def get_flu_coca_d(self):

        count = 0
        set_of_words = set()

        for item in self.get_normalized_tokens():
            set_of_words.add(item)

        for k in set_of_words:
            count = count + reuters_distribution[k]

        try: return count / float(len(set_of_words))
        except ZeroDivisionError as e:
            # print(e)
            return 0

    @staticmethod
    def least_common_values(my_array, to_find=None):

        counter = collections.Counter(my_array)
        if to_find is None:
            return sorted(counter.items(), key=itemgetter(1), reverse=False)
        return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))

# sentences
# big words
# syllables per word
# short sentences
# long sentences
# Lexical Diversity	   try:             return float(len(text_tokens_dictionary)) / float(sum(text_tokens_dictionary.values()))
# Content Word Diversity	float(len(text_tokens_dictionary)) / float(sum(text_tokens_dictionary.values()))
# Redundancy	return float(my_count) / float(nlp.get_filtered_words_count())
# Typographical error ratio	float(nlp.get_errors()) / float(ts.get_words())
def gen_nlp_features(text):
    nlp = MyBasicNLP(text)
    results = {}
    results['error'] = nlp.get_errors()
    results['long_sent'] = nlp.get_long_sentences()
    results['short_sent'] = nlp.get_short_sentences()
    results['flu_coca_c'] = nlp.get_flu_coca_c()
    results['flu_coca_d'] = nlp.get_flu_coca_d()
    results['stop_wd_p'] = nlp.stop_words_percent()
    return results

def test_gen_pos_features():
    text = "Republican presidential candidate Ohio Gov. John Kasich speaks at a campaign event Wednesday, March 23, in Wauwatosa, Wis. | AP Photo When Kasich is matched up against Clinton, 45 percent of registered voters nationwide said they would vote for the Ohio governor, compared to 39 percent for the former secretary of state. Against Clinton, Kasich leads with men, voters between the ages of 18 and 54and white non-Hispanics, while Clinton holds a narrower advantage among women and a wider lead among those who are not white. Even Lindsey Graham, who has endorsed Cruz despite their own colorful history of disagreements, said Thursday that Kasich would be the best presidential candidate and a better president. ""I think John Kasich would be the best nominee, but he doesn't have a chance,"" Graham said on MSNBC's ""Morning Joe,"" adding, ""John Kasich's problem is he is an insider in an outsider year, and nobody seems to want to buy that."" In a theoretical three-way race between Trump, Clinton and Libertarian candidate Gary Johnson, Clinton earned 42 percent, Trump 34 percent and Johnson 11 percent. The Monmouth poll also suggests that Kasich and Bernie Sanders are the only candidates still in the race who are seen more favorably than not. Whereas Clinton had a net negative rating of -11 points (40 percent positive to 51 percent negative) and Trump is far below at -30 points (30 percent to 60 percent), Kasich's favorability sits at +32 points (50 percent to 18 percent), though 32 percent said they had no opinion of him. Approval of Cruz, while still a net negative 6 points (37 percent to 43 percent), has risen 8 points since October and 12 points since last June. Monmouth conducted its poll March 17-20 via landlines and cellphones, surveying 1,008 adults nationwide, including 848 registered voters. For that subsample, the margin of error is plus or minus 3.4 percentage points."
    result = gen_nlp_features(text)
    print(len(result), result)
    return

if __name__ == '__main__':
    test_gen_pos_features()
    pass