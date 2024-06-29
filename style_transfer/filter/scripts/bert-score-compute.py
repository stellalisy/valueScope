import os
import gc
import random
import pandas as pd
import pickle
import torch
import json
import argparse
import math
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import logging
import lime
import numpy as np
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace, Vocabulary
import seaborn as sns
from evaluate import load
import matplotlib.pyplot as plt
import ast
from scipy.stats import pearsonr
from bert_score import score
random.seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddit", type=str)
    parser.add_argument("--dimension", type=str)
    return parser.parse_args()
args = get_args()

# Directories
BERTSCORE_FORMATTED_DIR = '/gscratch/argon/hjung10/norm_discovery_project/code/style_transfer_filter/content_preservation/bertscore-generated-formatted/'
NATURALNESS_FILTER_DIR = '/gscratch/argon/hjung10/norm_discovery_project/code/style_transfer_filter/filters/naturalness_filter/passed_filter/'

# this function gathers existing IDs in a set that passed the filter so we don't need to append
# those IDs again in our files within PATH. This 
def read_existing_dictionary(PATH):
    reddit_to_data = dict()
    for file in os.listdir(PATH):
        if ".jsonl" in file:
            set_data = set()
            with open(PATH + file, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    set_data.update(line.keys())   # adding the keys (e.g. comment IDs)

            reddit_to_data[file] = set_data
    return reddit_to_data

# this function gathers existing IDs as well as their mapped values in a dictionary that passed the filter
def read_existing_dictionary_key_value(PATH):
    reddit_to_data = dict()
    for file in os.listdir(PATH):
        if ".jsonl" in file:
            data_dict = dict()
            with open(PATH + file, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    data_dict.update(line)     # adding the ID to comments together

            reddit_to_data[file] = data_dict
    return reddit_to_data

# PATH and file_name of the file to append to for list_dict
# previous_reddit_to_data represents the existing data so we don't append data that's already in the file
def save_jsonl(PATH, file_name, list_dict, previous_reddit_to_data):
    new_added = 0
    with open(PATH + file_name, 'a') as f:
        for dict_ in list_dict:
            if file_name not in previous_reddit_to_data or (file_name in previous_reddit_to_data and list(dict_.keys())[0] not in previous_reddit_to_data[file_name]):
                new_added += 1
                json.dump(dict_, f)
                f.write('\n')
    return new_added

# determine new bertscore computations needed
def determine_new_bertscores_needed(reddit_to_ppl_data, bertscore_computed):
    reddit_to_compute_needed = dict()

    for reddit_json, dict_synthetic in reddit_to_ppl_data.items():
        # contains set of synthetic comment IDs whose perplexity have already been computed
        comments_computed_set = None
        if reddit_json in bertscore_computed:
            comments_computed_set = bertscore_computed[reddit_json]

        # dictionary to keep track of comments that we need to compute perplexity for
        dict_compute = dict() 
        int_computed = 0
        for synthetic_id, tuple_comments in dict_synthetic.items():
            if comments_computed_set == None or synthetic_id not in comments_computed_set:
                dict_compute[synthetic_id] = tuple_comments
            else:
                int_computed += 1
        print(reddit_json)
        print("Number of synthetic comments needed for compute: " + str(len(dict_compute)))
        print("Number of synthetic comments already computed: " + str(int_computed))
        reddit_to_compute_needed [reddit_json] = dict_compute
        print()
    return reddit_to_compute_needed
    
# loading bertscore library
bertscore = load("bertscore")
subreddit_target = args.subreddit

# reading in prior style transfer generations to avoid computing the perplexity for new generations
bertscore_computed = read_existing_dictionary(BERTSCORE_FORMATTED_DIR)
reddit_to_ppl_data = read_existing_dictionary_key_value(NATURALNESS_FILTER_DIR)
print("Loaded data")

subreddit = args.subreddit
dimension = args.dimension
file_name = subreddit + '_' + dimension + '.jsonl'

reddit_to_compute_needed = determine_new_bertscores_needed(reddit_to_ppl_data, bertscore_computed)
print(subreddit)
print(dimension)
print(len(reddit_to_compute_needed[file_name]))
print("-------------")

total_added = 0
list_synthetic_id_to_bertscore = []
for i, (synthetic_id, list_comment) in enumerate(reddit_to_compute_needed[file_name].items()):
    if i % 10000 == 9999:
        added = save_jsonl(BERTSCORE_FORMATTED_DIR, file_name, list_synthetic_id_to_bertscore, dict())
        list_synthetic_id_to_bertscore = []
        total_added += added
        print(i)
        print("Saving dictionary")
      
    synthetic_comment = list_comment[0]
    original_comment = list_comment[1]
    synthetic_id_to_bertscore = dict()
    try:
        results = bertscore.compute(predictions=[synthetic_comment], references=[original_comment], model_type='microsoft/deberta-xlarge-mnli')
        synthetic_id_to_bertscore[synthetic_id] = results
        list_synthetic_id_to_bertscore.append(synthetic_id_to_bertscore)  
    except Exception as e:
        print(e)
        print(synthetic_comment)
        print(original_comment)
        gc.collect()
        torch.cuda.empty_cache()
        continue
added = save_jsonl(BERTSCORE_FORMATTED_DIR, file_name, list_synthetic_id_to_bertscore, dict())
total_added += added
print("Total added: " + str(total_added))