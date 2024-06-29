import os
import gc
import random
import pandas as pd
import pickle
import torch
import json
import math
from collections import defaultdict
import logging
import lime
import seaborn as sns
from evaluate import load
import argparse
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace, Vocabulary
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
#import transformers

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium")
    parser.add_argument("--subreddit", type=str)
    parser.add_argument("--dimension", type=str)
    return parser.parse_args()
args = get_args()

PERPLEXITY_FORMATTED_DIR = '/gscratch/argon/hjung10/norm_discovery_project/code/style_transfer_filter/naturalness/dialoGPT_formatted/'
LINGUISTIC_FILTER_DIR = '/gscratch/argon/hjung10/norm_discovery_project/code/style_transfer_filter/filters/lexical_filter/passed_filter/'

# loading the model
compute_dtype = getattr(torch, "float16")
model_name = 'microsoft/DialoGPT-medium'

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
access_token = #insert your own token here
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=compute_dtype, device_map="auto", token=access_token)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print("Loaded model")

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

def compute_perplexity(model, tokenizer, text):
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs['labels'] = inputs['input_ids'].detach().clone().to("cuda" if torch.cuda.is_available() else "cpu")

    # Get the logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
        
    avg_loss = outputs.loss
    perplexities = torch.exp(avg_loss).tolist()

    return perplexities


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

# first step is to compute perplexity values for any new style transfer generations
def determine_new_style_transfers_needed(reddit_to_lexical_data, perplexity_computed):
    reddit_to_compute_needed = dict()
    
    for reddit_json, dict_synthetic in reddit_to_lexical_data.items():
        # contains set of synthetic comment IDs whose perplexity have already been computed
        comments_computed_set = None
        if reddit_json in perplexity_computed:
            comments_computed_set = perplexity_computed[reddit_json]
    
        # dictionary to keep track of comments that we need to compute perplexity for
        dict_compute = dict() 
        int_computed = 0
        for synthetic_id, tuple_comments in dict_synthetic.items():
    
            # the perplexity for the synthetic comment has not been computed; add to compute list
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

# reading in prior style transfer generations to avoid computing the perplexity for new generations
perplexity_computed = read_existing_dictionary(PERPLEXITY_FORMATTED_DIR)
reddit_to_lexical_data = read_existing_dictionary_key_value(LINGUISTIC_FILTER_DIR)
print("Loaded data")

subreddit = args.subreddit
dimension = args.dimension
file_name = subreddit + '_' + dimension + '.jsonl'

reddit_to_compute_needed = determine_new_style_transfers_needed(reddit_to_lexical_data, perplexity_computed)
print(subreddit)
print(dimension)
print(len(reddit_to_compute_needed[file_name]))
print("-------------")

total_added = 0
list_synthetic_id_to_perplexity = []
for i, (synthetic_id, list_comment) in enumerate(reddit_to_compute_needed[file_name].items()):
    # periodically save data
    if i % 20000 == 19999:
        added = save_jsonl(PERPLEXITY_FORMATTED_DIR, file_name, list_synthetic_id_to_perplexity, dict())
        list_synthetic_id_to_perplexity = []
        total_added += added
        print(i)
        print("Saving dictionary")
      
    synthetic_comment = list_comment[0]
    synthetic_id_to_perplexity = dict()
    try:
        perplexity = compute_perplexity(model, tokenizer, synthetic_comment)
        synthetic_id_to_perplexity[synthetic_id] = perplexity
        list_synthetic_id_to_perplexity.append(synthetic_id_to_perplexity)  
    except:
        print(synthetic_comment)
added = save_jsonl(PERPLEXITY_FORMATTED_DIR, file_name, list_synthetic_id_to_perplexity, dict())
total_added += added
print("Total added: " + str(total_added))