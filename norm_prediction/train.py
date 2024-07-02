import os
import gc
import sys
import json
import random
import pandas as pd
import pickle
import torch
import logging
import transformers
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Determine the device to be used for computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define constants
MAX_LEN = 1024
MODEL_NAME = "microsoft/deberta-v3-base"
str_modelname = MODEL_NAME.split("/")[-1]

# Define paths and parameters (use capitalized variables for paths)
PATH_DATA = "YOUR_DATA_PATH/"
CACHE_DIR = "YOUR_CACHE_DIR/"
PATH_SAVE = "YOUR_SAVE_PATH/"

# Get parameters from command line arguments
NORM_DIMENSION = sys.argv[1]
CATEGORY = sys.argv[2]
assert NORM_DIMENSION in ['supportiveness', 'sarcasm', 'politeness', 'humor', 'formality'], f"{NORM_DIMENSION} should be one of five dimensions"
assert CATEGORY in ['politics', 'gender', 'finance', 'science', 'askscience']

SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Function to initialize model and tokenizer
def initialize_model(modelname, bool_tokenizer=False, bool_model=True):
    tokenizer = AutoTokenizer.from_pretrained(modelname, truncation=True, max_len=MAX_LEN, padding='max_length', cache_dir=CACHE_DIR) if bool_tokenizer else None
    model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=2, cache_dir=CACHE_DIR).to(device) if bool_model else None
    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model and "DialoGPT" in modelname:
        model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

# Function to process input text
def process_input_text(d, reverse_label=False):
    input1 = d['post_title_1'] + "\n" + d['post_description_1'] if isinstance(d['post_description_1'], str) else d['post_title_1']
    input2 = d['post_title_2'] + "\n" + d['post_description_2'] if isinstance(d['post_description_2'], str) else d['post_title_2']
    if reverse_label:
        return f"Comment 1: {d['comment_2']}\nComment 2: {d['comment_1']}\nPOST1: {input2}\nPOST2: {input1}"
    else:
        return f"Comment 1: {d['comment_1']}\nComment 2: {d['comment_2']}\nPOST1: {input1}\nPOST2: {input2}"

# Function to process data
def process_data(data, balance_label=True):
    if data is None:
        return None
    processed = []
    num_labels = [0, 0]
    for _, d in data.iterrows():
        if 'annotation' in d:
            try:
                label = int(d['annotation'][NORM_DIMENSION]) - 1
            except:
                print(f"label not either 1 or 2 (label: {d['annotation'][NORM_DIMENSION]})")
                continue
        elif 'synthetic_label' in d:
            try:
                label = int(d['synthetic_label']) - 1
            except:
                print(f"label not either 1 or 2")
                print(d['synthetic_label'])
                continue
        if balance_label:
            less_num_label = 0 if num_labels[0] < num_labels[1] else 1
            if less_num_label == label:
                input_text = process_input_text(d)
                instance = [input_text, less_num_label]
            else:
                input_text = process_input_text(d, reverse_label=True)
                instance = [input_text, less_num_label]
            num_labels[less_num_label] += 1
        else:
            input_text = process_input_text(d)
            instance = [input_text, label]
        tokenized = tokenizer(instance[0], max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
        tokenized = {'input_ids': tokenized['input_ids'][0], 'attention_mask': tokenized['attention_mask'][0]}
        if 'token_type_ids' in tokenized:
            tokenized['token_type_ids'] = tokenized['token_type_ids'][0]
        instance[0] = tokenized
        processed.append(instance)
    return processed

# Custom dataset class
class CustomData(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Initialize tokenizer
_, tokenizer = initialize_model(MODEL_NAME, bool_tokenizer=True, bool_model=False)

# Load and preprocess data
with open(os.path.join(PATH_DATA, f"{CATEGORY}_binary.pkl"), "rb") as file:
    raw_data = pickle.load(file)
raw_data = pd.DataFrame(raw_data).rename({
    'title1': 'post_title_1', 'description1': 'post_description_1', 'comment1': 'comment_1',
    'title2': 'post_title_2', 'description2': 'post_description_2', 'comment2': 'comment_2'}, axis=1)

synthetic = pd.read_csv(os.path.join(PATH_DATA, f"{CATEGORY}_supportive_toxic_synthetic_data.csv")).rename({
    'title1': 'post_title_1', 'description1': 'post_description_1', 'comment1': 'comment_1',
    'title2': 'post_title_2', 'description2': 'post_description_2', 'comment2': 'comment_2'}, axis=1)

synthetic_old = None
if NORM_DIMENSION == "supportiveness" and CATEGORY == "gender":
    synthetic_old = pd.read_csv(os.path.join(PATH_DATA, "gender_supportive_toxic_synthetic_data_old.csv")).rename({
        'title1': 'post_title_1', 'description1': 'post_description_1', 'comment1': 'comment_1',
        'title2': 'post_title_2', 'description2': 'post_description_2', 'comment2': 'comment_2'}, axis=1)

# Process data
data_human = process_data(raw_data)
data_synthetic = process_data(synthetic)
data_synthetic_old = process_data(synthetic_old)

# Define data loaders
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_RATIO = 0.2
VAL_RATIO = 0.1

if data_synthetic_old is not None:
    data_synthetic_and_old = data_synthetic + data_synthetic_old
    random.shuffle(data_synthetic_and_old)
    num_val = int(len(data_synthetic_and_old) * VAL_RATIO)
    data_dev, data_train = data_synthetic_and_old[:num_val], data_synthetic_and_old[num_val:]
    dataloader_train = DataLoader(CustomData(data_train), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    dataloader_dev = DataLoader(CustomData(data_dev), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(CustomData(data_human), batch_size=VALID_BATCH_SIZE, shuffle=True)
else:
    random.shuffle(data_synthetic)
    num_val = int(len(data_synthetic) * VAL_RATIO)
    data_dev, data_train = data_synthetic[:num_val], data_synthetic[num_val:]
    dataloader_train = DataLoader(CustomData(data_train), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    dataloader_dev = DataLoader(CustomData(data_dev), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(CustomData(data_human), batch_size=VALID_BATCH_SIZE, shuffle=True)

# Function to evaluate model
def evaluate(dataloader):
    model.eval()
    test_loss, num_example, num_correct = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input, lbl = batch
            input = {k: v.to(device) for k, v in input.items()}
            lbl = lbl.to(device)
            outputs = model(**input, labels=lbl)
            test_loss += outputs.loss
            num_example += len(lbl)
            _, pred_idx = outputs.logits.max(dim=1)
            num_correct += sum(pred_idx == lbl).item()
            all_preds.extend(pred_idx.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='micro')
    return test_loss / len(dataloader), num_correct / num_example, f1

# Function to train model
def train(epoch=0, best_val_acc=0, best_test_acc=0, best_test_loss=0, best_val_f1=0, best_test_f1=0, bool_verbose=True, bool_save=True, min_save_epoch=3):
    model.train()
    train_loss = 0
    for _, batch in enumerate(dataloader_train):
        input, lbl = batch
        input = {k: v.to(device) for k, v in input.items()}
        lbl = lbl.to(device)
        outputs = model(**input, labels=lbl)
        loss = outputs.loss
        train_loss += loss
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    if bool_verbose:
        print(f"Train Loss: {train_loss / len(dataloader_train)}")
    val_loss, val_acc, val_f1 = evaluate(dataloader_dev)
    test_loss, test_acc, test_f1 = evaluate(dataloader_test)
    print(f"val_loss: {val_loss}, val_acc: {val_acc}, val_f1: {val_f1}")
    print(f"test_loss: {test_loss}, test_acc: {test_acc}, test_f1: {test_f1}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_test_acc = test_acc
        best_test_loss = test_loss
        best_test_f1 = test_f1
        if bool_save:
            if epoch >= min_save_epoch:
                torch.save(model.state_dict(), os.path.join(PATH_SAVE, f"{str_modelname}-{CATEGORY}-{NORM_DIMENSION}-epoch{epoch}.pth"))
                with open(os.path.join(PATH_SAVE, f"optimizer-{str_modelname}-{CATEGORY}-{NORM_DIMENSION}-epoch{epoch}.pkl"), "wb") as file:
                    pickle.dump(optimizer, file)
                with open(os.path.join(PATH_SAVE, f"scheduler-{str_modelname}-{CATEGORY}-{NORM_DIMENSION}-epoch{epoch}.pkl"), "wb") as file:
                    pickle.dump(scheduler, file)
    return val_acc, best_val_acc, best_val_f1, best_test_acc, best_test_loss, best_test_f1

# Initialize model and optimizer
model, _ = initialize_model(MODEL_NAME, bool_tokenizer=False, bool_model=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=20 * len(dataloader_train))

# Run training loop
EPOCHS = 20
best_val_acc, best_val_f1, best_test_acc, best_test_loss, best_test_f1 = 0, 0, 0, 0, 0
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")
    val_acc, best_val_acc, best_val_f1, best_test_acc, best_test_loss, best_test_f1 = train(epoch, best_val_acc, best_test_acc, best_test_loss, best_val_f1, best_test_f1, bool_verbose=True)
    print(f"best_val_acc: {best_val_acc}, best_val_f1: {best_val_f1}")
    print(f"best_test_acc: {best_test_acc}, best_test_loss: {best_test_loss}, best_test_f1: {best_test_f1}")

# Log best metrics
logging.info(f"best_val_acc: {best_val_acc}, best_val_f1: {best_val_f1}")
logging.info(f"best_test_acc: {best_test_acc}, best_test_loss: {best_test_loss}, best_test_f1: {best_test_f1}")
