import os
import gc
import sys
import json
import random
import pickle
import math
import more_itertools
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from itertools import combinations
from tqdm import tqdm
from collections import Counter

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)

TARGET_DIMENSION = sys.argv[1]
TARGET_TOPIC = sys.argv[2]
assert TARGET_DIMENSION in ['supportiveness', 'sarcasm', 'politeness', 'humor', 'formality']
assert TARGET_TOPIC in ['gender', 'politics', 'science', 'finance']

MAX_LEN = 1024
MODEL_NAME = "microsoft/deberta-v3-base"

# Define your own paths here
BASE_PATH_MODELS = "<YOUR_BASE_PATH_TO_MODELS>"
BASE_PATH_OUTPUT = "<YOUR_BASE_PATH_TO_OUTPUT>"
BASE_PATH_COMMENTS = "<YOUR_BASE_PATH_TO_COMMENTS>"

PATH_MODELS = {
    "gender": {
        "formality": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_GENDER_FORMALITY>"),
        "politeness": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_GENDER_POLITENESS>"),
        "supportiveness": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_GENDER_SUPPORTIVENESS>"),
        "sarcasm": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_GENDER_SARCASM>"),
        "humor": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_GENDER_HUMOR>"),
    },
    "politics": {
        "formality": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_PILITICS_FORMALITY>"),
        "politeness": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_PILITICS_POLITENESS>"),
        "supportiveness": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_PILITICS_SUPPORTIVENESS>"),
        "sarcasm": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_PILITICS_SARCASM>"),
        "humor": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_PILITICS_HUMOR>"),
    },
    "science": {
        "formality": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_SCIENCE_FORMALITY>"),
        "politeness": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_SCIENCE_POLITENESS>"),
        "supportiveness": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_SCIENCE_SUPPORTIVENESS>"),
        "sarcasm": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_SCIENCE_SARCASM>"),
        "humor": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_SCIENCE_HUMOR>"),
    },
    "finance": {
        "formality": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_FINANCE_FORMALITY>"),
        "politeness": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_FINANCE_POLITENESS>"),
        "supportiveness": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_FINANCE_SUPPORTIVENESS>"),
        "sarcasm": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_FINANCE_SARCASM>"),
        "humor": os.path.join(BASE_PATH_MODELS, "<MODEL_PATH_FINANCE_HUMOR>"),
    }
}
PATH_MODEL = PATH_MODELS[TARGET_TOPIC][TARGET_DIMENSION]
PATH_OUT = os.path.join(BASE_PATH_OUTPUT, f"{TARGET_TOPIC}-{TARGET_DIMENSION}.jsonl")

BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 64
BOOL_PRIORITIZE_LESS = True if len(sys.argv) > 5 and sys.argv[5] == "True" else False
PRIORITY_CUTOFF = 30
SAVE_EVERY = 20
RELOAD_EVERY = 5000

TARGET_SUBREDDITS = {
    "gender": set(['askmen', 'askwomen', 'asktransgender']),
    "politics": set(['democrats', 'republican', 'libertarian']),
    "science": set(['askscience', 'shittyaskscience', 'asksciencediscussion']),
    "finance": set(['pennystocks', 'stocks', 'wallstreetbets', 'wallstreetbetsnew'])
}

BOOL_FILTER = True if len(sys.argv) > 4 and sys.argv[4] == "True" else False
PATH_FILTER = f"<YOUR_PATH_TO_FILTER>/ids2keep-content_preservation-{TARGET_TOPIC}-{TARGET_DIMENSION}.json"

if os.path.isfile(PATH_FILTER) and BOOL_FILTER:
    with open(PATH_FILTER, "r") as file:
        comments2keep = json.load(file)
    comments2keep = set(comments2keep)
else:
    comments2keep = None

FILTER_PHRASES = ['I apologize, but', 'not able to fulfill this request', 'cannot fulfill your request', 'I cannot provide a rewritten', 'rewritten comment']
SPLIT_PHRASES = ['"\n\nThis', '"\n\nRating:', '"\n\nPlease', '"\n\nNote', '"\n\nRewritten', '"\n\nOriginal', '"\n\nRemember', '"\n\nI rewrote', '"\n\nI hope', '"\n\nI rated', '"\n\nI Changed', '"\n\nI kept', '"\n\nI tried', '"\n\nThe original', '"\n\nI\'ve']

def filter_comment(comment):
    if any(phrase in comment for phrase in FILTER_PHRASES):
        return None
    for split_phrase in SPLIT_PHRASES:
        if split_phrase in comment:
            return comment.split(split_phrase)[0]
    return comment

def process_data(df, subreddit):
    _data = []
    for _, row in df.iterrows():
        original_id = row['id'].replace("t1_", "")
        if comments2keep and original_id+"-0" not in comments2keep:
            continue
        _data.append({'id': original_id+"-0", 'id-original': original_id, 'comment': row['original_comment'], 'rating': row['original_rating'], 'post_title': row['submission_title']})
        for scale in ['1', '2', '3', '4', '5']:
            filtered_comment = filter_comment(row[scale])
            if filtered_comment is not None:
                _data.append({'id': original_id+"-"+scale, 'id-original': original_id, 'comment': filtered_comment, 'rating': int(scale), 'post_title': row['submission_title']})
    return _data

def process_input(d1, d2):
    return f"Comment 1: {d1['comment']}\nComment 2: {d2['comment']}\n\nQuestion: Does comment 2 show more {TARGET_DIMENSION} than comment 1? Yes or No?"

# Model Initialization
print(f"Load tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Load model from {PATH_MODEL}...")
model = AutoModelForSequenceClassification.from_pretrained(PATH_MODEL).to(DEVICE)
print("Done.")

# Preparing Data
if os.path.isfile(PATH_OUT):
    with open(PATH_OUT, "r") as file:
        scored_pairs = json.load(file)
    completed_pairs = set(scored_pairs.keys())
    start_idx = len(scored_pairs)
else:
    scored_pairs = {}
    completed_pairs = set()
    start_idx = 0

comments_df = []
print("Loading dataframes...")
for path in [os.path.join(BASE_PATH_COMMENTS, subreddit + ".csv") for subreddit in TARGET_SUBREDDITS[TARGET_TOPIC]]:
    comments_df.append(pd.read_csv(path))
comments_df = pd.concat(comments_df)

print("Processing dataframes...")
original_data = []
for subreddit in TARGET_SUBREDDITS[TARGET_TOPIC]:
    original_data += process_data(comments_df, subreddit)

print("Data processing done.")

# Model Inference
num_batches = math.ceil(len(original_data) / BATCH_SIZE)
for batch_num in tqdm(range(start_idx, num_batches)):
    batch_data = original_data[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]
    input_texts = [process_input(d1, d2) for d1, d2 in combinations(batch_data, 2)]
    inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

    for i, (d1, d2) in enumerate(combinations(batch_data, 2)):
        pair_id = f"{d1['id']}_{d2['id']}"
        scored_pairs[pair_id] = {
            "input_text": input_texts[i],
            "score": scores[i],
            "id1": d1['id-original'],
            "id2": d2['id-original']
        }

    if (batch_num + 1) % SAVE_EVERY == 0:
        with open(PATH_OUT, "w") as file:
            json.dump(scored_pairs, file)
        print(f"Saved at batch {batch_num + 1}.")

print("Inference completed.")
