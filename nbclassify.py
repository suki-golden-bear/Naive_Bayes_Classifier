#Author: Suki Sahota
import math
from nblearn import NBLearn

class NBClassify:
    global_vocab = set()
    cardinality_of_vocab = 0
    
    pos_map = {}
    tru_map = {}
    neg_map = {}
    dec_map = {}
    
    num_pos = 0
    num_tru = 0
    num_neg = 0
    num_dec = 0

    pos_mode_val = 0
    tru_mode_val = 0
    neg_mode_val = 0
    dec_mode_val = 0

    @staticmethod
    def set_learned_variables(master_dict):
        NBClassify.global_vocab = set(master_dict["global_vocab"])
        NBClassify.cardinality_of_vocab = len(NBClassify.global_vocab)

        NBClassify.pos_map = master_dict["pos_map"]
        NBClassify.tru_map = master_dict["tru_map"]
        NBClassify.neg_map = master_dict["neg_map"]
        NBClassify.dec_map = master_dict["dec_map"]

        NBClassify.num_pos = master_dict["num_pos"]
        NBClassify.num_tru = master_dict["num_tru"]
        NBClassify.num_neg = master_dict["num_neg"]
        NBClassify.num_dec = master_dict["num_dec"]

        NBClassify.pos_mode_val = master_dict["pos_mode_val"]
        NBClassify.tru_mode_val = master_dict["tru_mode_val"]
        NBClassify.neg_mode_val = master_dict["neg_mode_val"]
        NBClassify.dec_mode_val = master_dict["dec_mode_val"]

    @staticmethod
    def process_unknown_review(f, test_map, num_test):
        for line in f:
            for raw_word in line.split():
                word = NBLearn.sanitize_word(raw_word.strip())
                if len(word) < 3:
                    continue
                if word not in NBLearn.STOP_WORDS:
                    #Add to given dict and increment its respective count
                    test_map[word] = test_map.get(word, 0) + 1
                    num_test = num_test + 1

        return test_map, num_test

    @staticmethod
    def classify_label_a(test_map, num_test):
        truthful_prob = NBClassify.find_truth_prob(test_map, num_test)
        deceptive_prob = NBClassify.find_decep_prob(test_map, num_test)
        return 'truthful' if truthful_prob > deceptive_prob else 'deceptive'

    @staticmethod
    def find_truth_prob(test_map, num_test):
        #For log-sum-exp trick
        max_a_k = math.log(
            (NBClassify.tru_mode_val+1) \
            /(NBClassify.num_tru+NBClassify.cardinality_of_vocab))

        tru_prob = max_a_k #Return value for truthful probability

        for term, times in test_map.items():
            #Need to continue if term is not in class map
            if term not in NBClassify.tru_map:
                continue
            freq = NBClassify.tru_map[term]
            p_hat = (freq+1) \
                /(NBClassify.num_tru+NBClassify.cardinality_of_vocab)
            a_k = math.log(p_hat)

            cur_prob = math.e**(a_k-max_a_k)
            tru_prob = tru_prob+(times*cur_prob)

        return tru_prob

    @staticmethod
    def find_decep_prob(test_map, num_test):
        #For log-sum-exp trick
        max_a_k = math.log(
            (NBClassify.dec_mode_val+1) \
            /(NBClassify.num_dec+NBClassify.cardinality_of_vocab))

        dec_prob = max_a_k #Return value for deceptive probability

        for term, times in test_map.items():
            #Need to continue if term is not in class map
            if term not in NBClassify.dec_map:
                continue
            freq = NBClassify.dec_map[term]
            p_hat = (freq+1) \
                /(NBClassify.num_dec+NBClassify.cardinality_of_vocab)
            a_k = math.log(p_hat)

            cur_prob = math.e**(a_k-max_a_k)
            dec_prob = dec_prob+(times*cur_prob)

        return dec_prob

    @staticmethod
    def classify_label_b(test_map, num_test):
        positive_prob = NBClassify.find_pos_prob(test_map, num_test)
        negative_prob = NBClassify.find_neg_prob(test_map, num_test)
        return 'positive' if positive_prob > negative_prob else 'negative'

    @staticmethod
    def find_pos_prob(test_map, num_test):
        #For log-sum-exp trick
        max_a_k = math.log(
            (NBClassify.pos_mode_val+1) \
            /(NBClassify.num_pos+NBClassify.cardinality_of_vocab))

        pos_prob = max_a_k #Return value for positive probability

        for term, times in test_map.items():
            #Need to continue if term is not in class map
            if term not in NBClassify.pos_map:
                continue
            freq = NBClassify.pos_map[term]
            p_hat = (freq+1) \
                /(NBClassify.num_pos+NBClassify.cardinality_of_vocab)
            a_k = math.log(p_hat)

            cur_prob = math.e**(a_k-max_a_k)
            pos_prob = pos_prob+(times*cur_prob)

        return pos_prob

    @staticmethod
    def find_neg_prob(test_map, num_test):
        #For log-sum-exp trick
        max_a_k = math.log(
            (NBClassify.neg_mode_val+1) \
            /(NBClassify.num_neg+NBClassify.cardinality_of_vocab))

        neg_prob = max_a_k #Return value for negative probability

        for term, times in test_map.items():
            #Need to continue if term is not in class map
            if term not in NBClassify.neg_map:
                continue
            freq = NBClassify.neg_map[term]
            p_hat = (freq+1) \
                /(NBClassify.num_neg+NBClassify.cardinality_of_vocab)
            a_k = math.log(p_hat)

            cur_prob = math.e**(a_k-max_a_k)
            neg_prob = neg_prob+(times*cur_prob)

        return neg_prob

###Driver code###
import glob
import json
import os
import sys

input_path = sys.argv[1]
master_dict = {}

#Grab information from model
with open('nbmodel.txt') as json_file:
    master_dict = json.load(json_file)

NBClassify.set_learned_variables(master_dict)

#Create nboutput.txt file
with open('nboutput.txt', 'w') as outfile:
    for filename in glob.iglob('./'+input_path+'/*/*/*/*.txt'):
        pathn = os.path.relpath(filename) #Print as third element in output
        with open(filename, 'r') as f:
            test_map = {}
            num_test = 0
            test_map, num_test = NBClassify.process_unknown_review(
                f, test_map, num_test)
            #Find truthful or deceptive first and save as label_a
            label_a = NBClassify.classify_label_a(test_map, num_test)
            #Find positive or negative second and save as label_b
            label_b = NBClassify.classify_label_b(test_map, num_test)
            print(label_a, label_b, pathn, file=outfile)
#################
