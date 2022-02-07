#Author: Suki Sahota
class NBLearn:
    #Set of stop words
    STOP_WORDS = { 'the', 'and', 'The', 'was', 'for', 'with', 'were', 'that' } 

    AMP = '&'
    CLOSING_ANG = '>'
    CLOSING_BRA = ']'
    CLOSING_PAR = ')'
    COLON = ':'
    COLON_SEMI = ';'
    COMMA = ','
    DASH = '-'
    DOT = '.'
    EQUAL = '='
    EXCLAMATION = '!'
    OPENING_ANG = '<'
    OPENING_BRA = '['
    OPENING_PAR = '('
    PLUS = '+'
    POUND = '#'
    QUESTION = '?'
    QUOTE_DOUBLE = '"'
    QUOTE_SINGLE = '\''
    SLASH = '/'
    STAR = '*'
    EXTRA_SYMBOLS = { AMP, CLOSING_ANG, CLOSING_BRA, CLOSING_PAR, COLON,
        COLON_SEMI, COMMA, DASH, DOT, EQUAL, EXCLAMATION, OPENING_ANG,
        OPENING_BRA, OPENING_PAR, PLUS, POUND, QUESTION, QUOTE_DOUBLE,
        QUOTE_SINGLE, SLASH, STAR }

    global_vocab = set() #List of entire vocabulary for all classes

    @staticmethod
    def process_known_review(f, first_map, num_first, first_mode_val,
        second_map, num_second, second_mode_val):
        for line in f:
            for raw_word in line.split():
                word = NBLearn.sanitize_word(raw_word.strip())
                if len(word) < 3:
                    continue
                if word not in NBLearn.STOP_WORDS:
                    NBLearn.global_vocab.add(word) #Add to global set
                    #Add to given dict and increment its respective count
                    first_map[word] = first_map.get(word, 0) + 1
                    first_mode_val = max(first_map[word], first_mode_val)
                    num_first = num_first + 1

                    second_map[word] = second_map.get(word, 0) + 1
                    second_mode_val = max(second_map[word], second_mode_val)
                    num_second = num_second + 1

        return num_first, num_second, first_mode_val, second_mode_val

    @staticmethod
    def sanitize_word(raw_word):
        #word = raw_word.lower() #I think case matters for deceptive reviews
        word = raw_word

        while len(word) > 0 and word[0] in NBLearn.EXTRA_SYMBOLS:
            word = word[1:]

        while len(word) > 0 and word[-1] in NBLearn.EXTRA_SYMBOLS:
            word = word[:-1]

        return word

###Driver code###
import glob
import json
import os
import sys

POS = 'positive_polarity'
POS_TRUTH = 'truthful_from_TripAdvisor'
POS_DECEP = 'deceptive_from_MTurk'
NEG = 'negative_polarity'
NEG_TRUTH = 'truthful_from_Web'
NEG_DECEP = 'deceptive_from_MTurk' #Same name as from positive, deceptive

#Maps to store key=word and value=frequency of word
pos_map = {} #Positive
tru_map = {} #Truthful
neg_map = {} #Negative
dec_map = {} #Deceptive

#Total number of non–stop words per classification
num_pos = 0
num_tru = 0
num_neg = 0
num_dec = 0

pos_mode_val = 0
tru_mode_val = 0
neg_mode_val = 0
dec_mode_val = 0

input_path = sys.argv[1]

#Train on Positive and Truthful reviews
for filename in glob.iglob(
    os.path.join(input_path, POS, POS_TRUTH, "fold?", "*.txt")):
    with open(filename, 'r') as f:
        num_pos, num_tru, pos_mode_val, tru_mode_val \
            = NBLearn.process_known_review(f, pos_map, num_pos, pos_mode_val,
                tru_map, num_tru, tru_mode_val)

#Train on Positive and Deceptive reviews
for filename in glob.iglob(
    os.path.join(input_path, POS, POS_DECEP, "fold?", "*.txt")):
    with open(filename, 'r') as f:
        num_pos, num_dec, pos_mode_val, dec_mode_val \
            = NBLearn.process_known_review(f, pos_map, num_pos, pos_mode_val,
                dec_map, num_dec, dec_mode_val)

#Train on Negative and Truthful reviews
for filename in glob.iglob(
    os.path.join(input_path, NEG, NEG_TRUTH, "fold?", "*.txt")):
    with open(filename, 'r') as f:
        num_neg, num_tru, neg_mode_val, tru_mode_val \
            = NBLearn.process_known_review(f, neg_map, num_neg, neg_mode_val,
                tru_map, num_tru, tru_mode_val)

#Train on Negative and Deceptive reviews
for filename in glob.iglob(
    os.path.join(input_path, NEG, NEG_DECEP, "fold?", "*.txt")):
    with open(filename, 'r') as f:
        num_neg, num_dec, neg_mode_val, dec_mode_val \
            = NBLearn.process_known_review(f, neg_map, num_neg, neg_mode_val,
                dec_map, num_dec, dec_mode_val)

pos_map = \
    dict(sorted(pos_map.items(), key=lambda item: item[1], reverse=True)[:100])
tru_map = \
    dict(sorted(tru_map.items(), key=lambda item: item[1], reverse=True)[:100])
neg_map = \
    dict(sorted(neg_map.items(), key=lambda item: item[1], reverse=True)[:100])
dec_map = \
    dict(sorted(dec_map.items(), key=lambda item: item[1], reverse=True)[:100])

#Combine model into one master dictionary
master_dict = {
    "global_vocab": list(NBLearn.global_vocab),
    "pos_map": pos_map,
    "tru_map": tru_map,
    "neg_map": neg_map,
    "dec_map": dec_map,
    "num_pos": num_pos,
    "num_tru": num_tru,
    "num_neg": num_neg,
    "num_dec": num_dec,
    "pos_mode_val": pos_mode_val,
    "tru_mode_val": tru_mode_val,
    "neg_mode_val": neg_mode_val,
    "dec_mode_val": dec_mode_val,
}

#Write as JSON object to .txt file
with open('./nbmodel.txt', 'w') as out_file:
    json.dump(master_dict, out_file, indent=4)
#################
