#!/usr/bin/env python
# coding: utf8
import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import textacy
    from textacy import extract
    from textacy import preprocess
    from textacy import constants
    import spacy
import string
import nltk
from nltk.parse import stanford
from nltk import parse

def extract_pattern_list(doc, pptn):
    return list(extract.pos_regex_matches(doc, pptn))
    # return list(extract.matches(doc, pptn))

my_nlp = spacy.load('en_core_web_sm')
class TextStatistics:
    """
    TextStats is based in Spacy and textacy
    in dependencies you should install en_core_web_sm for english language
    """
    def __init__(self, text):
        # self.text = preprocess.fix_bad_unicode(text=text, normalization='NFC')
        self.text = text
        self.doc = my_nlp(self.text)
        self.ts = textacy.text_stats.TextStats(self.doc)
        self.marks = lambda l1: sum([1 for x in l1 if x == '!'])
        self.punctuation = lambda l1, l2: sum([1 for x in l1 if x in l2])
        self.quotes = lambda l1: sum([1 for x in l1 if x == '"'])
        self.tokenizer = my_nlp.Defaults.create_tokenizer()
        self.tokens = self.tokenizer(self.text)

    def get_text(self):
        return self.text

    def get_tokens(self):
        return self.tokens

    def noun_phrases(self):
        """

        :return:
        """
        pattern = constants.POS_REGEX_PATTERNS['en']['NP']
        return extract_pattern_list(self.doc, pattern)

    def noun_phrases_count(self):
        """

        :return:
        """
        return len(self.noun_phrases())

    def get_noun_phrases_words(self):
        """

        :return:
        """
        counts = []
        [counts.append(len(noun_phrase)) for noun_phrase in self.noun_phrases()]
        return sum(counts)

    def noun_phrase_avg_length(self):
        """
        In noun phrase tokens is included punctuation

        :return: float
        """
        try:
            return float(self.get_noun_phrases_words()) / float(len(self.noun_phrases()))
        except ZeroDivisionError as e:
            # print("Noun phrase error: ", e)
            return 0

    def verb_phrases(self):
        """

        :return:
        """
        pattern = constants.POS_REGEX_PATTERNS['en']['VP']
        return extract_pattern_list(self.doc, pattern)

    def verb_phrases_count(self):
        """

        :return:
        """
        return len(self.verb_phrases())

    def get_verb_phrases_words(self):

        counts = []
        [counts.append(len(verb_phrase)) for verb_phrase in self.verb_phrases()]
        return sum(counts)

    def verb_phrase_avg_length(self):

        try:
            return float(self.get_verb_phrases_words()) / float(len(self.verb_phrases()))
        except ZeroDivisionError as e:
            # print("Noun phrase error: ", e)
            return 0

    def get_clauses(self):
        """

        :return:
        """
        pattern = textacy.constants.POS_REGEX_PATTERNS['en']['NP']
        pattern = pattern + '<VERB>'
        lst = extract_pattern_list(self.doc, pattern)
        return len(lst)

    def get_flesh_kincaid(self):
        try:
            return self.ts.flesch_kincaid_grade_level
        except ZeroDivisionError as e:
            # print(e)
            return 0

    def get_fog_index(self):
        try:
            return self.ts.gunning_fog_index
        except ZeroDivisionError as e:
            # print("Fog Index error: ", e)
            return 0

    def get_smog_index(self):
        try:
            return self.ts.smog_index
        except ZeroDivisionError as e:
            # print("Smog Index: ", e)
            return 0

    def get_basic_counts(self):
        return self.ts.basic_counts

    def get_long_words(self):
        return self.ts.basic_counts['n_long_words']

    def get_chars(self):
        return self.ts.basic_counts['n_chars']

    def get_monosyllable_words(self):
        return self.ts.basic_counts['n_monosyllable_words']

    def get_polysyllable_words(self):
        return self.ts.basic_counts['n_polysyllable_words']

    def get_sentences(self):
        return self.ts.basic_counts['n_sents']

    def get_syllables(self):
        return self.ts.basic_counts['n_syllables']

    def get_unique_words(self):
        return self.ts.basic_counts['n_unique_words']

    def get_words(self):
        return self.ts.basic_counts['n_words']

    def get_average_syllables_per_word(self):
        """

        :return:
        """
        sylls = self.get_syllables()
        words = self.get_words()
        try:
            return float(sylls) / float(words)
        except ZeroDivisionError:
            return 0

    def get_average_words_per_sentence(self):
        """

        :return:
        """
        words = self.get_words()
        sents = self.get_sentences()
        try:
            return float(words) / float(sents)
        except ZeroDivisionError:
            return 0

    def get_exclamation_marks(self):
        """
        Needs improvement. Maybe Spacy provides such function
        :return:
        """
        return self.marks(self.text)

    def get_punctuation(self):
        """
        Needs improvement. Maybe Spacy provides such function
        :return:
        """
        return self.punctuation(self.text, set(string.punctuation))

    def get_quotes(self):
        """
        Needs improvement. Maybe Spacy provides such function
        :return:
        """
        return self.quotes(self.text)

    def get_capital_words(self):
        return sum([1 for word in self.tokens if str(word).isupper()])

    def get_average_word_length(self):
        try:
            return float(self.get_chars())/float(self.get_words())
        except ZeroDivisionError:
            return 0

    def get_pausality(self):
        try:
            return float(self.get_punctuation())/float(self.get_sentences())
        except ZeroDivisionError as e:
            # print("pausality caused error: ", e)
            return 0

    def get_lexical_word_diversity(self):
        try:
            return float(self.get_unique_words())/float(self.get_words())
        except ZeroDivisionError as e:
            # print("Lexical word diversity: ",e)
            return 0

#todo his features
# # Words	Just a word count
# # Verbs	Verb count (I used pos tags)
# # Noun Phrases	I used regular expressions for that one
# # Sentences	Spacy textacy text stats
# avg # clauses	def get_clauses(self):  pattern = textacy.constants.POS_REGEX_PATTERNS['en']['NP']  pattern = pattern + '<VERB>'  return len(list(textacy.extract.pos_regex_matches(self.doc, pattern)))	devided by the number of words
# avg sentence length	Use Spacy - textacy statistics for that
# avg word length	Use Spacy - textacy statistics for that
# avg noun phrase length	Use Spacy - textacy statistics for that
# Pausality	float(self.get_punctuation())/float(self.get_sentences())
# Flesh Kincaid grade level
def gen_nltk_features(text):
    ts = TextStatistics(text)
    basic_counts = ts.get_basic_counts()
    basic_counts['noun_phrase_words'] = ts.get_noun_phrases_words()
    basic_counts['verb_phrase_words'] = ts.get_verb_phrases_words()
    basic_counts['noun_phrase_count'] = ts.noun_phrases_count()
    basic_counts['verb_phrase_count'] = ts.verb_phrases_count()
    basic_counts['ave_clause'] = ts.get_clauses()
    basic_counts['ave_word_per_sent'] = ts.get_average_words_per_sentence()
    basic_counts['ave_word_len'] = ts.get_average_word_length()
    basic_counts['ave_np_len'] = ts.noun_phrase_avg_length()
    basic_counts['ave_vp_len'] = ts.verb_phrase_avg_length()
    basic_counts['pausality'] = ts.get_pausality()
    basic_counts['flesh_kincaid'] = ts.get_flesh_kincaid()
    # print(basic_counts)
    return basic_counts

def test_gen_nltk_features():
    text = "Republican presidential candidate Ohio Gov. John Kasich speaks at a campaign event Wednesday, March 23, in Wauwatosa, Wis. | AP Photo When Kasich is matched up against Clinton, 45 percent of registered voters nationwide said they would vote for the Ohio governor, compared to 39 percent for the former secretary of state. Against Clinton, Kasich leads with men, voters between the ages of 18 and 54and white non-Hispanics, while Clinton holds a narrower advantage among women and a wider lead among those who are not white. Even Lindsey Graham, who has endorsed Cruz despite their own colorful history of disagreements, said Thursday that Kasich would be the best presidential candidate and a better president. ""I think John Kasich would be the best nominee, but he doesn't have a chance,"" Graham said on MSNBC's ""Morning Joe,"" adding, ""John Kasich's problem is he is an insider in an outsider year, and nobody seems to want to buy that."" In a theoretical three-way race between Trump, Clinton and Libertarian candidate Gary Johnson, Clinton earned 42 percent, Trump 34 percent and Johnson 11 percent. The Monmouth poll also suggests that Kasich and Bernie Sanders are the only candidates still in the race who are seen more favorably than not. Whereas Clinton had a net negative rating of -11 points (40 percent positive to 51 percent negative) and Trump is far below at -30 points (30 percent to 60 percent), Kasich's favorability sits at +32 points (50 percent to 18 percent), though 32 percent said they had no opinion of him. Approval of Cruz, while still a net negative 6 points (37 percent to 43 percent), has risen 8 points since October and 12 points since last June. Monmouth conducted its poll March 17-20 via landlines and cellphones, surveying 1,008 adults nationwide, including 848 registered voters. For that subsample, the margin of error is plus or minus 3.4 percentage points."
    result = gen_nltk_features(text)
    print(len(result), result)
    return

if __name__ == '__main__':
    # test_gen_nltk_features()
    pass