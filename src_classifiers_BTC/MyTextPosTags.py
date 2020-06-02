import nltk
from collections import Counter


class MyTextPosTags:

    """
    based on NLTK default TagSet
    http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    https://catalog.ldc.upenn.edu/docs/LDC99T42/tagguid1.pdf

    1. Coordinating conjunctions


    """

    def __init__(self, my_tokens):
        self.my_tokens = my_tokens

    def pos_tags_counter(self):
        """

        :return: dictionary
        """
        word_tag = nltk.pos_tag(self.my_tokens)
        counts = Counter(tag for word, tag in word_tag)
        return counts

    def get_pos_tag(self, tag):
        return self.pos_tags_counter()[tag]

    def get_coordinating_conjunctions(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['CC']

    def get_cardinal_numbers(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['CD']

    def get_determiners(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['DT']

    def get_existential_there(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['EX']

    def get_foreign_words(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['FW']

    def get_subordinating_conjunctions(self):
        """

        :return:
        """
        return self.pos_tags_counter()['IN']

    def get_adjectives(self):
        """
        JJ: Hyphenated compounds that are
            used as modifiers are tagged as adjectives
        :return: int
        """
        return self.pos_tags_counter()['JJ']

    def get_adjectives_comparative(self):
        """
        JJR: Adjectives with the comparative
        ending -er and a comparative meaning
        :return: int
        """
        return self.pos_tags_counter()['JJR']

    def get_adjectives_superlative(self):
        """
        JJS: Adjectives with the superlative ending -est
        :return:
        """
        return self.pos_tags_counter()['JJS']

    def get_item_markers(self):
        """

        :return:
        """
        return self.pos_tags_counter()['LS']

    def get_modal(self):
        """
         This category includes all verbs that
         don't take an -s ending in the third person
         singular present. e.g. (can, could, dare,
         may, might, must, ought, shall, should, will,
         would)

        :return: int
        """
        return self.pos_tags_counter()['MD']

    def get_noun_singular(self):
        """


        :return: int
        """
        return self.pos_tags_counter()['NN']

    def get_noun_plural(self):
        """

        :return:
        """
        return self.pos_tags_counter()['NNS']

    def get_proper_noun_singular(self):
        """

        :return:
        """
        return self.pos_tags_counter()['NNP']

    def get_proper_noun_plural(self):
        """

        :return:
        """
        return self.pos_tags_counter()['NNPS']

    def get_predeterminer(self):
        """

        :return:
        """
        return self.pos_tags_counter()['PDT']

    def get_possessive_ending(self):
        """

        :return:
        """
        return self.pos_tags_counter()['POS']

    def get_personal_pronouns(self):
        """

        :return:
        """
        return self.pos_tags_counter()['PRP']

    def get_possessive_pronouns(self):
        """

        :return:
        """
        return self.pos_tags_counter()['PRP$']

    def get_adverbs(self):
        """
        RB: this category includes most words that end
        in -ly as well as degree words like quite, too
        and very
        :return: int
        """
        return self.pos_tags_counter()['RB']

    def get_adverbs_comparative(self):
        """
        RBR: Adverbs with the comparative ending -er
        :return: int
        """
        return self.pos_tags_counter()['RBR']

    def get_adverbs_superlative(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['RBS']

    def get_particles(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['RP']

    def get_symbols(self):
        """
    
        :return: int
        """
        return self.pos_tags_counter()['SYM']

    def get_to(self):
        """

        :return:
        """
        return self.pos_tags_counter()['TO']

    def get_interjection(self):
        """

        :return:
        """
        return self.pos_tags_counter()['UH']

    def get_verb(self):
        """

        :return:
        """
        return self.pos_tags_counter()['VB']

    def get_verb_past_tense(self):
        """

        :return:
        """
        return self.pos_tags_counter()['VBD']

    def get_verb_gerund(self):
        """
            counts the occurrences of verbs
            in gerund or present participle
            forms
        :return: int
        """
        return self.pos_tags_counter()['VBG']

    def get_verb_past_participle(self):
        """
            counts the occurrences of verbs
            in past participle forms
        :return: int
        """
        return self.pos_tags_counter()['VBN']

    def get_verb_non_third_person(self):
        """
            counts the occurrences of verbs
            in non-3rd person singular present
        :return: int
        """
        return self.pos_tags_counter()['VBP']

    def get_verb_third_person(self):
        """
            counts the occurrences of verbs
            in 3rd person singular present
            forms
        :return: int
        """
        return self.pos_tags_counter()['VBZ']

    def get_wh_determiner(self):
        """
            counts the occurrences of wh-determiners
        :return: int
        """
        return self.pos_tags_counter()['WDT']

    def get_wh_pronoun(self):
        """
            counts the occurrences of wh-pronouns
        :return: int
        """
        return self.pos_tags_counter()['WP']

    def get_possessive_wh_pronoun(self):
        """
            counts the occurrences of  possessive
            wh-pronouns
        :return: int
        """
        return self.pos_tags_counter()['WP$']

    def get_wh_adverb(self):
        """
            counts the occurrences of wh-averbs
        :return: int
        """
        return self.pos_tags_counter()['WRB']

    def get_rate_of_adjectives_adverbs(self):
        try:
            return float(self.pos_tags_counter()['JJ']) / self.pos_tags_counter()['RB']
        except ZeroDivisionError as e:
            # print("Raa returned ",e)
            return 0

    def get_emotiveness(self):
        # (total # of adjectives + total # of adverbs) / (total # of noun + total # of verbs)
        nominator = self.pos_tags_counter()['JJ'] + self.pos_tags_counter()['RB']
        denominator = self.pos_tags_counter()['NN'] + self.pos_tags_counter()['NNS'] + self.pos_tags_counter()['VB']
        try:
            return float(nominator)/denominator
        except ZeroDivisionError as e:
            # print("Emotiveness returned ", e)
            return 0

    def get_modifiers(self):
        return self.get_adjectives() + self.get_adverbs()


# Modifiers	self.get_adjectives() + self.get_adverbs()
# # Modal Verbs	pos tags
# Emotiveness	nominator = self.pos_tags_counter()['JJ'] + self.pos_tags_counter()['RB'] denominator = self.pos_tags_counter()['NN'] + self.pos_tags_counter()['NNS'] + self.pos_tags_counter()['VB']  try:   return float(nominator)/denominator
from nltk import word_tokenize
def gen_pos_features(text):
    tokens = word_tokenize(text)
    mtpt = MyTextPosTags(tokens)
    result = mtpt.pos_tags_counter()
    rate_adj_adv = mtpt.get_rate_of_adjectives_adverbs()
    emotive = mtpt.get_emotiveness()
    result['modifiers'] = rate_adj_adv
    result['emotive'] = emotive
    return result

def test_gen_pos_features():
    text = "Republican presidential candidate Ohio Gov. John Kasich speaks at a campaign event Wednesday, March 23, in Wauwatosa, Wis. | AP Photo When Kasich is matched up against Clinton, 45 percent of registered voters nationwide said they would vote for the Ohio governor, compared to 39 percent for the former secretary of state. Against Clinton, Kasich leads with men, voters between the ages of 18 and 54and white non-Hispanics, while Clinton holds a narrower advantage among women and a wider lead among those who are not white. Even Lindsey Graham, who has endorsed Cruz despite their own colorful history of disagreements, said Thursday that Kasich would be the best presidential candidate and a better president. ""I think John Kasich would be the best nominee, but he doesn't have a chance,"" Graham said on MSNBC's ""Morning Joe,"" adding, ""John Kasich's problem is he is an insider in an outsider year, and nobody seems to want to buy that."" In a theoretical three-way race between Trump, Clinton and Libertarian candidate Gary Johnson, Clinton earned 42 percent, Trump 34 percent and Johnson 11 percent. The Monmouth poll also suggests that Kasich and Bernie Sanders are the only candidates still in the race who are seen more favorably than not. Whereas Clinton had a net negative rating of -11 points (40 percent positive to 51 percent negative) and Trump is far below at -30 points (30 percent to 60 percent), Kasich's favorability sits at +32 points (50 percent to 18 percent), though 32 percent said they had no opinion of him. Approval of Cruz, while still a net negative 6 points (37 percent to 43 percent), has risen 8 points since October and 12 points since last June. Monmouth conducted its poll March 17-20 via landlines and cellphones, surveying 1,008 adults nationwide, including 848 registered voters. For that subsample, the margin of error is plus or minus 3.4 percentage points."
    tokens = word_tokenize(text)
    result = gen_pos_features(tokens)
    print(len(result), result)
    return

if __name__ == '__main__':
    # test_gen_pos_features()
    pass