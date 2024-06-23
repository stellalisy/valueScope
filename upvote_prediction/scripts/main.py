import torch
import json
from datasets import Dataset
import numpy as np
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import sklearn
import pickle
from datetime import datetime
import argparse
import wandb
import csv
from collections import defaultdict
import pandas as pd
from distutils.dir_util import copy_tree

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subreddit", type=str, default="askmen")
    parser.add_argument("-d", "--dimension", type=str, default="formality")
    parser.add_argument("-w", "--wandb_name", type=str, default=None)
    parser.add_argument("--topic", type=str, default="gender")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("-l", "--label_type", type=str, default="label_log")
    parser.add_argument("-e", "--sample_eval_size", type=int, default=5000)
    parser.add_argument("-m", "--model_name", type=str, default=None)
    parser.add_argument("-t", "--sample_train_size", type=int, default=0)
    parser.add_argument("--train_on_first_half", action="store_true")
    parser.add_argument("-r", "--root_dir", type=str, default="/gscratch/argon/stelli")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--total_batch_size", type=int, default=128)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("--time", type=str, default=None)
    parser.add_argument("--author", type=str, default=None)
    parser.add_argument("--nopost", action="store_true")
    parser.add_argument("--max_score_range", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="/gscratch/argon/chanyoun/reddit-norms/style_transfer")
    parser.add_argument("--output_dir", type=str, default="/gscratch/argon/stelli/reddit_norm/upvote_prediction/data")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_parent_dir", type=str, default="/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models")
    parser.add_argument("--error_alpha", type=float, default=0.1)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--run_key", type=str, default=None)
    parser.add_argument("--processed_data_dir", type=str, default="/gscratch/argon/stelli/reddit_norm/upvote_prediction/data/processed_upvotes")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100000000)
    return parser.parse_args()
args = get_args()
subreddit = args.subreddit
print("subreddit: ", subreddit)
if args.wandb_name is None: args.wandb_name = args.subreddit
if args.wandb_group is None: args.wandb_group = args.subreddit
if args.model_name is None: args.model_name = f"{subreddit}_dialogueRPT"

default_model_dir = {
    "askmen": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/askmen_dialogueRPT_log_half_2ep_time",
    "askwomen": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/askwomen_dialogueRPT_log_half_2ep_time",
    "asktransgender": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/asktransgender_dialogueRPT_log_half_5ep_time",
    "askscience": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/askscience_dialogueRPT_log_half_5ep_time",
    "asksciencediscussion": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/asksciencediscussion_dialogueRPT_log_half_5ep_time",
    "shittyaskscience": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/shittyaskscience_dialogueRPT_log_half_5ep_time",
    "democrats": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/democrats_dialogueRPT_log_half_5ep_time",
    "republican": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/republican_dialogueRPT_log_half_5ep_time",
    "libertarian": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/libertarian_dialogueRPT_log_half_2ep_time_author",
    "wallstreetbets": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/wallstreetbets2m_dialogueRPT_log_2m_2ep_time",
    "stocks": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/stocks_dialogueRPT_log_half_5ep_time",
    "pennystocks": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/pennystocks_dialogueRPT_log_half_5ep_time",
    "wallstreetbetsnew": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/wallstreetbetsnew_dialogueRPT_log_half_5ep_time"
}
if args.model_dir == None and args.subreddit in default_model_dir:
    if not os.path.exists(os.path.join(default_model_dir[args.subreddit], "config.json")):
        newest_checkpoint = 0
        for dir in os.listdir(default_model_dir[args.subreddit]):
            if 'checkpoint-' in dir and os.path.exists(os.path.join(default_model_dir[args.subreddit], dir, "config.json")) and f'-{args.run_key}' in dir:
                args.model_dir = os.path.join(default_model_dir[args.subreddit], dir)
                break
            if 'checkpoint-' in dir and os.path.exists(os.path.join(default_model_dir[args.subreddit], dir, "config.json")) and dir.split('-')[-1].isdigit() and int(dir.split('-')[-1]) > newest_checkpoint:
                newest_checkpoint = int(dir.split('-')[-1])
        if args.model_dir == None:
            source_dir = os.path.join(default_model_dir[args.subreddit], f"checkpoint-{newest_checkpoint}")
            copy_tree(source_dir, f"{source_dir}-{args.run_key}")
            args.model_dir = f"{source_dir}-{args.run_key}"
    else:
        args.model_dir = default_model_dir[args.subreddit]
print("model_dir: ", args.model_dir)
if args.predict:
    if "time" in args.model_dir: args.time = "readable"
    if "author" in args.model_dir: args.author = "all"


def seconds_to_time_string(seconds):
    # Calculate the days, hours, minutes, and seconds
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    
    # Create a readable format
    result = []
    if days: result.append(f"{days} days")
    if hours: result.append(f"{hours} hours")
    if minutes: result.append(f"{minutes} minutes")
    if sec: result.append(f"{sec} seconds")
    
    # Join all parts and return
    return " ".join(result)

def preprocess_function(examples):
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    for title, comment, label, post_time, reply_time, author, author_id in zip (examples["submission_title"], examples["comment"], examples[args.label_type], examples["created_submission"], examples["created_comment"], examples["author"], examples["author_id"]):
        if args.nopost: input_text = f"{comment}"
        else: input_text = f"{title} || {comment}"
        
        if args.time == "unix": input_text = f"Time of reply: {reply_time}, which is {reply_time-post_time} after post. || {input_text}"
        elif args.time == "string_old": input_text = f"{input_text} || Reply time after post: {reply_time-post_time}. "
        elif args.time == "readable": input_text = f"Time of reply: {datetime.fromtimestamp(reply_time).strftime('%A %B %d %Y %H:%M:%S %p')}, which is {seconds_to_time_string(reply_time-post_time)} after post. || {input_text}"
        
        if args.author != None: input_text = f"Author: {author} ({author_id}) || {input_text}"
        
        tokenized = tokenizer(input_text, truncation=True, padding='max_length', max_length=MAX_LENGTH)
        new_examples["input_ids"].append(tokenized['input_ids'])
        new_examples["attention_mask"].append(tokenized['attention_mask'])
        new_examples["labels"].append(label)
    
    return new_examples

def bucket_extreme(label, max_range=200):
    if label > max_range: return max_range
    if label < -max_range: return -max_range
    return label

def load_data(data, seed, sample_train_size=0, sample_eval_size=0, max_range=200):
    # data = [d for d in data if abs(d['label']) < max_range and abs(d['label']) > -max_range]
    labels = np.array([bucket_extreme(d['label'], max_range=max_range) for d in data])
    label_mean = labels.mean()
    label_std = np.std(labels)
    print("mean", label_mean, 'std', label_std)
    for d in data:
        d['label_z'] = (d['label']-label_mean)/label_std
        d['label_log'] = np.log2(d['label']) if d['label'] > 0 else -np.log2(-d['label']) if d['label'] < 0 else 0.0
        d['label_norm'] = d['label']/label_mean

    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=0.1 if sample_eval_size < 1 else sample_eval_size*2, seed=seed)
    train, dev_test = dataset['train'], dataset['test']
    dataset_dev_test = dev_test.train_test_split(test_size=0.5 if sample_eval_size < 1 else sample_eval_size, seed=seed)
    dev, test = dataset_dev_test['train'], dataset_dev_test['test']

    original_columns = train.column_names
    if sample_train_size>0 and len(train)>sample_train_size:
        print("sampled {} samples from train dataset of size {}".format(sample_train_size, len(train)))
        train = train.select(range(sample_train_size))
    # if sample_eval_size>0 and len(test)>sample_eval_size:
    #     print("sampled {} samples from eval dataset of size {}".format(sample_eval_size, len(test)))
    #     test = test.select(range(sample_eval_size))
    train = train.map(preprocess_function, batched=True, num_proc=NUM_PROC, remove_columns=original_columns)
    # train = train.filter(lambda x: len(x["input_ids"]) <= 512)
    dev = dev.map(preprocess_function, batched=True, num_proc=NUM_PROC, remove_columns=original_columns)
    test = test.map(preprocess_function, batched=True, num_proc=NUM_PROC, remove_columns=original_columns)
    # test = test.filter(lambda x: len(x["input_ids"]) <= 512)
    print("train dataset: ", len(train), "dev dataset: ", len(dev), "test dataset: ", len(test))
    return train, dev, test, label_mean, label_std

def load_inference_data(data, label_mean=None, label_std=None, max_range=200, datasize=0):
    data = data[:datasize] if datasize > 0 else data
    labels = np.array([bucket_extreme(d['label'], max_range=max_range) for d in data])
    if label_mean == None: label_mean = labels.mean()
    if label_std == None: label_std = np.std(labels)
    for d in data:
        d['label_z'] = (d['label']-label_mean)/label_std
        d['label_log'] = np.log2(d['label']) if d['label'] > 0 else -np.log2(-d['label']) if d['label'] < 0 else 0.0
        d['label_norm'] = d['label']/label_mean

    dataset = Dataset.from_list(data)
    original_columns = dataset.column_names
    test = dataset.map(preprocess_function, batched=True, num_proc=min(NUM_PROC, len(data)), remove_columns=original_columns)
    # print("test dataset: ", len(test))
    return test

class RegressionTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')
            loss = torch.mean(torch.square(logits.squeeze() - labels.squeeze()))
            return (loss, outputs) if return_outputs else loss

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred #.float32()
    labels = labels.reshape(-1, 1) #.float32()
    # print("logits: ", logits)
    # print(f"numpy.isnan(logits).any(): {np.isnan(logits).any()}")
    # print(f"numpy.isnan(labels).any(): {np.isnan(labels).any()}")
    if np.any(np.isnan(logits)):
        print("logits: ", logits)
    if np.any(np.isnan(labels)):
        print("labels: ", labels)

    mse = sklearn.metrics.mean_squared_error(labels, logits)
    mae = sklearn.metrics.mean_absolute_error(labels, logits)
    r2 = sklearn.metrics.r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()

    # Compute accuracy
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracies, nums = pairwise_accuracy(logits, labels, gaps=[0.1, 0.3, 1.0, 0.01])
    accuracy = accuracies[0]
    accuracy3 = accuracies[1] if len(accuracies) > 1 else 0
    accuracy1 = accuracies[2] if len(accuracies) > 2 else 0
    accuracy01 = accuracies[3] if len(accuracies) > 3 else 0
    num_samples = nums[0]
    num_samples3 = nums[1] if len(nums) > 1 else 0
    num_samples1 = nums[2] if len(nums) > 2 else 0
    num_samples01 = nums[3] if len(nums) > 3 else 0

    out = {"mse": mse, "mae": mae, "r2": r2, 
           "accuracy": accuracy, "accuracy3": accuracy3, "accuracy1": accuracy1, "accuracy01": accuracy01,
           "num_samples": num_samples, "num_samples3": num_samples3, "num_samples1": num_samples1, "num_samples01": num_samples01}

    return out

def pairwise_accuracy(logits, labels, gaps=[0.1]):
    corrects = {gap: [] for gap in gaps}
    # print("logits: ", logits)
    for i in range(len(logits)):
        for j in range(i+1, len(logits)):
            for min_gap in gaps:
                if abs(labels[i] - labels[j]) < min_gap: continue
                if (logits[i] > logits[j]) != (labels[i] > labels[j]): corrects[min_gap].append(0)
                else: corrects[min_gap].append(1)
    out = [sum(corrects[min_gap])/len(corrects[min_gap]) if len(corrects[min_gap]) > 0 else 0 for min_gap in gaps]
    nums = [len(corrects[min_gap]) for min_gap in gaps]
    return out, nums

def pairwise_accuracy_inference(logits, labels, ids, gaps=[0.1]):
    corrects = {gap: defaultdict(list) for gap in gaps}
    # print("logits: ", logits)
    for i in range(len(logits)):
        for j in range(i+1, len(logits)):
            for min_gap in gaps:
                if abs(labels[i] - labels[j]) < min_gap: continue
                if (logits[i] > logits[j]) != (labels[i] > labels[j]): 
                    corrects[min_gap][ids[i]].append(0)
                    corrects[min_gap][ids[j]].append(0)
                    corrects[min_gap]['all'].append(0)
                else: 
                    corrects[min_gap][ids[i]].append(1)
                    corrects[min_gap][ids[j]].append(1)
                    corrects[min_gap]['all'].append(1)
    for gap in gaps:
        corrects[gap] = {k: (sum(v)/len(v),len(v)) if len(v) > 0 else 0 for k, v in corrects[gap].items()}
    return corrects

def foo_metrics(eval_pred):
    return {"accuracy": 0}


NUM_PROC = 8
MAX_LENGTH = 512
TOTAL_BATCH_SIZE=args.total_batch_size

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print("device: ", device)

model_card = "microsoft/DialogRPT-updown"   # you can try other model_card listed in the table above
tokenizer = AutoTokenizer.from_pretrained(model_card)


def main_train_upvote_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_card)
    model.to(device)

    with open(f"{args.processed_data_dir}/{subreddit}.pkl", "rb") as file:
        data = pickle.load(file)
    data_ids = [d['idx'] if 'idx' in d else i for i, d in enumerate(data)]

    new_model_dir = f"{args.model_parent_dir}/{args.model_name}"
    if not os.path.isdir(new_model_dir):
        os.makedirs(new_model_dir)

    if args.train_on_first_half: 
        np.random.seed(42)
        rand_idx = np.random.permutation(len(data))
        rand_idx_first_half = rand_idx[:len(data)//2]
        rand_idx_second_half = rand_idx[len(data)//2:]
        data = [data[i] for i in rand_idx_first_half]
        with open(os.path.join(new_model_dir, "train_sample_idxs.txt"), "w") as file:
            file.write('\n'.join([str(i) for i in rand_idx_first_half]))
        with open(os.path.join(new_model_dir, "inference_sample_idxs.txt"), "w") as file:
            file.write('\n'.join([str(i) for i in rand_idx_second_half]))
        with open(os.path.join(new_model_dir, "train_sample_ids.txt"), "w") as file:
            file.write('\n'.join([str(idx) for idx in [data_ids[i] for i in rand_idx_first_half]]))
        with open(os.path.join(new_model_dir, "inference_sample_ids.txt"), "w") as file:
            file.write('\n'.join([str(idx) for idx in [data_ids[i] for i in rand_idx_second_half]]))

    train_dataset, eval_dataset, test, label_mean, label_std = load_data(data, seed=42, sample_eval_size=args.sample_eval_size, max_range=args.max_score_range)
    with open(os.path.join(new_model_dir, "label_distribution.json"), "w") as file:
        file.write(json.dumps({'mean': label_mean, 'std': label_std}))
    print("loaded data")

    per_device_batch_size = args.batch_size
    gradient_accumulation_steps = TOTAL_BATCH_SIZE // per_device_batch_size

    run = wandb.init(project="reddit_upvote", group=args.subreddit, name=args.wandb_name, job_type="train")

    training_args = TrainingArguments(
            output_dir=new_model_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=0.001,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,  # 500,
            save_strategy="steps",
            save_steps=args.eval_steps,  # 500,
            save_total_limit=2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=False,
            # deepspeed=script_args.deepspeed,
            # local_rank=script_args.local_rank,
            remove_unused_columns=False,
            label_names=["labels"],
            # bf16=script_args.bf16,
            # fp16=True, #! this is important! if True, cuda out of memory.
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            optim="adamw_hf",
            lr_scheduler_type="linear",
            metric_for_best_model="accuracy",
            load_best_model_at_end=True,
        )

    trainer = RegressionTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_for_regression,
        )

    if any(['checkpoint-' in d for d in os.listdir(new_model_dir)]):
        trainer.train(resume_from_checkpoint = True)
    else:
        trainer.train()

    model.save_pretrained(new_model_dir)


def main_filter_original_comments():
    accuracy_threshold = 1 - args.error_alpha

    new_model_dir = args.model_dir
    if new_model_dir == None: new_model_dir = f"{args.root_dir}/reddit_norm/upvote_prediction/upvote_prediction_models/{args.model_name}"
    if not os.path.isdir(new_model_dir): raise ValueError("model_dir does not exist")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    per_device_batch_size = args.batch_size
    training_args = TrainingArguments(output_dir=new_model_dir, per_device_eval_batch_size=per_device_batch_size)
    trainer = RegressionTrainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics=foo_metrics)

    with open(os.path.join(new_model_dir, "label_distribution.json"), "r") as file:
        label_stats = json.load(file)
    label_mean, label_std = label_stats["mean"], label_stats["std"]

    
    # load and deduplicate the predictions
    if not os.path.exists(os.path.join(new_model_dir, f"predictions.csv")):
        with open(os.path.join(new_model_dir, f"predictions.csv"), "a+") as file:
            file.write("id,idx,prediction,label,acc_0.1,acc_0.3,acc_1.0,acc_0.01,total_0.1,total_0.3,total_1.0,total_0.01\n")
        predicted_idxs = []
    predictions_old = pd.read_csv(os.path.join(new_model_dir, f"predictions.csv"), index_col="id")
    predictions_old = predictions_old.loc[:, ~predictions_old.columns.str.contains('^Unnamed')]
    predictions_old.drop_duplicates(inplace=True)
    predictions_old.to_csv(os.path.join(new_model_dir, f"predictions.csv"))
    predicted_idxs = predictions_old['idx'].tolist()

    predictions_only = pd.read_csv(new_model_dir + '/predictions_only.csv') if os.path.exists(new_model_dir + '/predictions_only.csv') else pd.DataFrame(columns=["id", "idx", "prediction", "label"])
    predictions_only = predictions_only.loc[:, ~predictions_only.columns.str.contains('^Unnamed')]
    # find the idxs that exist in predcitions but not in predictions_only, and add them to predictions_only
    idxs_to_add = predictions_old[~predictions_old['id'].isin(predictions_only['id'])].index
    predictions_only = pd.concat([predictions_only, predictions_old.iloc[idxs_to_add][["id", "idx", "prediction", "label"]]])
    predictions_only.drop_duplicates(inplace=True)
    predictions_only.to_csv(new_model_dir + '/predictions_only.csv')

    with open(f"{args.processed_data_dir}/{subreddit}.pkl", "rb") as file:
        data = pickle.load(file)
    with open(os.path.join(new_model_dir, "inference_sample_idxs.txt"), "r") as file:
        rand_idx_second_half = [int(d) for d in file.read().split('\n') if d]
    if 'id' not in data[0]:
        with open(f"/gscratch/argon/stelli/reddit_norm/style_transfer/data/idx_to_id/{subreddit}_idx_to_id.pkl", "rb") as file:
            idx_to_id = pickle.load(file)
    inf_data = []
    for i in rand_idx_second_half:
        if i in predicted_idxs: continue
        if 'id' not in data[i]: data[i]['id'] = idx_to_id[i]
        temp = {'idx': i}
        temp.update(data[i])
        inf_data.append(temp)


    gaps = [0.1, 0.3, 1.0, 0.01]
    accumulative_acc = defaultdict(int)
    for gap in gaps:
        if os.path.exists(os.path.join(new_model_dir, f"filtered_ids_gap_{gap}_acc.txt")): 
            with open(os.path.join(new_model_dir, f"filtered_ids_gap_{gap}_acc.txt"), "r") as file:
                lines = file.readlines()
            accumulative_acc[gap] += sum([float(l.split('\t')[1]) for l in lines[1:]]) * args.sample_eval_size

    for i in range(0, len(inf_data), args.sample_eval_size):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}] Starting Ids: {i}-{min(i+args.sample_eval_size, len(inf_data))}")
        
        batch_new = inf_data[i:min(i+args.sample_eval_size, len(inf_data))]
        batch_new_idxs = [sample['idx'] for sample in batch_new]
        batch_old_idxs = []
        batch_old_pred = pd.DataFrame()

        corrects = {gap: defaultdict(list) for gap in gaps}
        if len(batch_new) > 0:
            inf_dataset = load_inference_data(batch_new, label_mean, label_std)

            predictions, labels, _ = trainer.predict(inf_dataset, metric_key_prefix="predict")
            print("Number of predictions generated:", len(predictions))
                
            corrects = pairwise_accuracy_inference(predictions, labels, batch_new_idxs, gaps=gaps)
            with open(os.path.join(new_model_dir, f"predictions.csv"), "a+") as file:
                # save the id, prediction, label, and accuracy on each line
                for j in range(len(predictions)):
                    res = {gap: corrects[gap][batch_new_idxs[j] if batch_new_idxs[j] in corrects[gap] else ('na','na')] for gap in gaps}
                    file.write(f"{batch_new[j]['id']},{batch_new_idxs[j]},{predictions[j][0]},{labels[j]},{res[0.1][0]},{res[0.3][0]},{res[1.0][0]},{res[0.01][0]},{res[0.1][1]},{res[0.3][1]},{res[1.0][1]},{res[0.01][1]}\n")
            with open(os.path.join(new_model_dir, f"predictions_only.csv"), "a+") as file:
                # save the id, prediction, label, and accuracy on each line
                for j in range(len(predictions)):
                    res = {gap: corrects[gap][batch_new_idxs[j] if batch_new_idxs[j] in corrects[gap] else ('na','na')] for gap in gaps}
                    file.write(f"{batch_new[j]['id']},{batch_new_idxs[j]},{predictions[j][0]},{labels[j]}\n")
        
        if len(batch_old_idxs) > 0:
            for gap in gaps:
                batch_old_gap = batch_old_pred.loc[:, [f'acc_{gap}',f'total_{gap}']]
                corrects[gap].update({key: tuple(batch_old_gap.loc[key].tolist()) for key in batch_old_idxs})
                num_correct = (corrects[gap]['all'][0] * corrects[gap]['all'][1] + sum([batch_old_gap.loc[key,f'acc_{gap}']*batch_old_gap.loc[key,f'total_{gap}'] for key in batch_old_idxs]))
                num_total = corrects[gap]['all'][1] + sum(batch_old_gap.loc[batch_old_idxs,f'total_{gap}'].tolist())
                corrects[gap]['all'] = (num_correct/num_total, num_total)

        filtered_ids = defaultdict(list)
        for gap in gaps:
            accumulative_acc[gap] += corrects[gap]['all'][0] * corrects[gap]['all'][1]
            print(f"Accuracy for gap {gap}: {corrects[gap]['all'][0]}, accumulative accuracy: {accumulative_acc[gap]/(i+len(predictions))}")
            filtered_ids[gap] = [k for k, v in corrects[gap].items() if v[0] > accuracy_threshold and k != 'all']
            print(f"Number of filtered ids for gap {gap}: ({len(filtered_ids[gap])/(len(corrects[gap])-1)}%) {len(filtered_ids[gap])}/{len(corrects[gap])-1}")
            with open(os.path.join(new_model_dir, f"filtered_ids_gap_{gap}.txt"), "a+") as file:
                file.write('\n'.join([str(idx) for idx in filtered_ids[gap]]) + '\n')
            with open(os.path.join(new_model_dir, f"filtered_ids_gap_{gap}_acc.txt"), "a+") as file:
                file.write(f"{i}\t{corrects[gap]['all'][0]}\t{accumulative_acc[gap]/(i+len(predictions))}\t{len(filtered_ids[gap])}\t{len(corrects[gap])-1}\t{len(filtered_ids[gap])/(len(corrects[gap])-1)}\n")
            


def main_predict_synthetic_comment():

    with open(f"{args.processed_data_dir}/{subreddit}.pkl", "rb") as file:
        raw_data = pickle.load(file)
    if 'id' not in raw_data[0]:
        with open(f"/gscratch/argon/stelli/reddit_norm/style_transfer/data/idx_to_id/{subreddit}_idx_to_id.pkl", "rb") as file:
            idx_to_id = pickle.load(file)
        raw_data = {idx_to_id[i]: d for i, d in enumerate(raw_data)}
    else:
        raw_data = {d['id'].replace("t1_", ''): d for i, d in enumerate(raw_data)}

    filename = os.path.join("/gscratch/argon/stelli/reddit_norm/style_transfer/data/output/llama3", f"{args.subreddit}_{args.dimension}.jsonl")
    with open(filename, "r") as json_file:
        json_list = list(json_file)
    inference_data = [json.loads(json_str) for json_str in json_list]

    for i in range(len(inference_data)):
        sample = {k: v.split(":\n\n")[-1] if type(v)==str else v for k, v in inference_data[i].items()}
        if 'id' in sample: sample['id'] = sample['id'].replace("t1_", '')
        inference_data[i] = sample
    print("loaded data from:", filename)
    print("number of inference data:", len(inference_data))

    new_model_dir = args.model_dir
    if new_model_dir == None: raise ValueError("model_dir is not provided")
    if not os.path.isdir(new_model_dir): raise ValueError("model_dir does not exist")
    
    with open(os.path.join(default_model_dir[args.subreddit], "label_distribution.json"), "r") as file:
        label_stats = json.load(file)
    label_mean, label_std = label_stats["mean"], label_stats["std"]
    

    # load and deduplicate the predictions for original (real) comments
    if not os.path.exists(os.path.join(new_model_dir, f"predictions.csv")):
        with open(os.path.join(new_model_dir, f"predictions.csv"), "w") as file:
            file.write("id,idx,prediction,label,acc_0.1,acc_0.3,acc_1.0,acc_0.01,total_0.1,total_0.3,total_1.0,total_0.01\n")
    predictions_old = pd.read_csv(os.path.join(new_model_dir, f"predictions.csv"))
    predictions_old.drop_duplicates(inplace=True)
    # predictions_old = predictions_old.loc[:, ~predictions_old.columns.str.contains('^Unnamed')]
    predictions_old.to_csv(os.path.join(new_model_dir, f"predictions.csv"))

    predictions_only = pd.read_csv(new_model_dir + '/predictions_only.csv') if os.path.exists(new_model_dir + '/predictions_only.csv') else pd.DataFrame(columns=["id", "idx", "prediction", "label"])
    # predictions_only = predictions_only.loc[:, ~predictions_only.columns.str.contains('^Unnamed')]
    # find the indices that exist in predcitions but not in predictions_only, and add them to predictions_only
    idxs_to_add = predictions_old[~predictions_old['id'].isin(predictions_only['id'])].index
    print("number of ids in predictions.csv but not predictions_only.csv:", len(idxs_to_add))
    predictions_only = pd.concat([predictions_only, predictions_old.iloc[idxs_to_add][["id", "idx", "prediction", "label"]]])
    
    # remove the rows with the idx value not in the inference data
    inferences_idxs = [sample['idx'] for sample in inference_data]
    predictions_only = predictions_only[predictions_only['idx'].isin(inferences_idxs)]
    print("number of ids in predictions_only.csv after filtering:", len(predictions_only))

    predictions_only.drop_duplicates(inplace=True)
    predicted_real_idxs = predictions_only['id'].tolist()

    predictions_only.to_csv(new_model_dir + '/predictions_only.csv')

    
    # read and deduplicate the output file
    out_filepath = os.path.join(args.output_dir, "upvote_predictions", f"predictions_{args.subreddit}_{args.dimension}.jsonl")
    if os.path.exists(out_filepath):
        with open(out_filepath, "r") as json_file: json_list = list(json_file)
        predicted_data = [json.loads(json_str) for json_str in json_list]
        print("loaded data from:", out_filepath)
        
        num_predicted_data = len(predicted_data)
        print("number of predicted data:", num_predicted_data)
        
        predicted_data = {sample['id']: sample for sample in predicted_data}
        predicted_syntetic_ids = list(predicted_data.keys())
        num_dedup_predicted_data = len(predicted_syntetic_ids)
        print("number of deduplicated predicted data:", num_dedup_predicted_data)
        
        if num_dedup_predicted_data < num_predicted_data:
            print(f"removed {num_predicted_data - num_dedup_predicted_data} duplicates from the predicted data")
            with open(out_filepath, "w") as f:
                for sample in predicted_data.values():
                    json.dump(sample, f)
                    f.write('\n')

    else: predicted_syntetic_ids = []

    num_inference_all = len(inference_data)
    print("Num inference data in the inference file:", num_inference_all)
    try: 
        inference_data = inference_data[args.start_idx:min(args.end_idx, len(inference_data))]
        print(f"Processing comments from index {args.start_idx} to {args.end_idx}")
    except: 
        pass
        print(f"Start & end range not specified / invalid, processing all remaining comments")
    num_in_range = len(inference_data)
    inference_data = [sample for sample in inference_data if sample['id'] not in predicted_syntetic_ids]
    print(f"Number of comments in the specified range not yet predicted: {len(inference_data)}/{num_in_range}")

    if len(inference_data) == 0:
        print("No comments to predict, exiting program")
        return
    
    pred_label_type = f'upvote_{args.label_type.split("_")[-1]}'

    model = AutoModelForSequenceClassification.from_pretrained(new_model_dir)
    model.to(device)
    print("Loaded model from:", new_model_dir)
    
    for global_i, sample in enumerate(inference_data):

        # if sample['id'] in predicted_syntetic_ids: 
        #     if raw_data[sample['id']]['created_comment'] > 1640998800:
        #         print(f"ID={sample['id']}, skipped, AFTER 2022-12-31")
        #     else:
        #         print(f"ID={sample['id']}, skipped")
        #     continue
        # if raw_data[sample['id']]['created_comment'] > 1640998800:
        #     print(f"ID={sample['id']}, AFTER 2022-12-31")
        
        if global_i % 100 == 0: 
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting at: {global_i}, Updating predicted_real_idxs")
            # predictions_only = pd.read_csv(new_model_dir + '/predictions_only.csv') if os.path.exists(new_model_dir + '/predictions_only.csv') else pd.DataFrame(columns=["id", "idx", "prediction", "label"])
            # predictions_only = predictions_only.loc[:, ~predictions_only.columns.str.contains('^Unnamed')]
            # predictions_only.drop_duplicates(inplace=True)
            # predicted_real_idxs = predictions_only['id'].tolist()

        if sample['id'] in predicted_real_idxs:
            new_id = f"{sample['id']}-0"
            pred = predictions_only.loc[sample['id']]['prediction']

            if 'log' in pred_label_type: orig_upvote = np.exp(pred)
            elif 'z' in pred_label_type: orig_upvote = pred * label_std + label_mean
            elif 'norm' in pred_label_type: orig_upvote = pred * label_mean
            else: orig_upvote = pred

            out_dict = {"id": sample["id"], "idx": sample["idx"], 
                        "original_rating": sample["original_rating"], "original_score": sample["original_score"], 
                        "submission_title": sample["submission_title"], 
                        'created_submission': raw_data[sample['id']]['created_submission'], 'created_comment': raw_data[sample['id']]['created_comment'], 
                        'author': raw_data[sample['id']]['author'], 'author_id': raw_data[sample['id']]['author_id'], 
                        new_id: {"comment": sample["original_comment"], pred_label_type: pred, 'upvote': orig_upvote}}
            # gather synthetic comments, predict, and then add to out_dict
            comments_to_predict = []
            for rating in ['1', '2', '3', '4', '5']:
                if rating in sample:
                    new_id = f"{sample['id']}-{rating}"
                    comments_to_predict.append({'idx': new_id,
                                        'comment': sample[rating],
                                        'submission_title': sample['submission_title'],
                                        'submission_body': '',
                                        'label': 0,
                                        'created_submission': raw_data[sample['id']]['created_submission'],
                                        'created_comment': raw_data[sample['id']]['created_comment'],
                                        'author': raw_data[sample['id']]['author'],
                                        'author_id': raw_data[sample['id']]['author_id']})
            predictions = model_predict(model, tokenizer, comments_to_predict, args.batch_size)
            # inf_dataset = load_inference_data(comments_to_predict, label_mean, label_std, datasize=args.sample_eval_size)
            # breakpoint()
            # predictions, labels, metrics = trainer.predict(inf_dataset, metric_key_prefix="predict")
            # breakpoint()
            for i in range(len(comments_to_predict)):
                comments_to_predict[i][pred_label_type] = float(predictions[i])
                comments_to_predict[i]['upvote'] = float(predictions[i]) * label_std + label_mean if "z" in args.label_type else np.exp(float(predictions[i]))
                out_dict[comments_to_predict[i]['idx']] = {'comment': comments_to_predict[i]['comment'], pred_label_type: comments_to_predict[i][pred_label_type], 'upvote': comments_to_predict[i]['upvote']}
            # append to output file
            with open(out_filepath, "a+") as f:
                json.dump(out_dict, f)
                f.write('\n')
            print(f"ID={sample['id']}: {pred} ({pred_label_type}), {predictions[0]}, {predictions[1]}, {predictions[2]}, {predictions[3]}, {predictions[4]}")
        else:
            out_dict = {"id": sample["id"], "idx": sample["idx"], "original_rating": sample["original_rating"], "original_score": sample["original_score"], "submission_title": sample["submission_title"], 'created_submission': raw_data[sample['id']]['created_submission'], 'created_comment': raw_data[sample['id']]['created_comment'], 'author': raw_data[sample['id']]['author'], 'author_id': raw_data[sample['id']]['author_id']}
            comments_to_predict = []

            if 'log' in pred_label_type: orig_label = np.log2(sample["original_score"]) if sample["original_score"] > 0 else -np.log2(-sample["original_score"]) if sample["original_score"] < 0 else 0
            elif 'z' in pred_label_type: orig_label = (sample["original_score"] - label_mean) / label_std
            elif 'norm' in pred_label_type: orig_label = sample["original_score"] / label_mean
            else: orig_label = sample["original_score"]

            for rating in ['0', '1', '2', '3', '4', '5']:
                new_id = f"{sample['id']}-{rating}"
                comments_to_predict.append({'idx': new_id,
                                    'comment': sample[rating] if rating != '0' else sample['original_comment'],
                                    'submission_title': sample['submission_title'],
                                    'submission_body': '',
                                    'label': 0 if rating != '0' else orig_label,
                                    'created_submission': raw_data[sample['id']]['created_submission'],
                                    'created_comment': raw_data[sample['id']]['created_comment'],
                                    'author': raw_data[sample['id']]['author'],
                                    'author_id': raw_data[sample['id']]['author_id']})
                breakpoint()
            predictions = model_predict(model, tokenizer, comments_to_predict, args.batch_size)
            for i in range(len(comments_to_predict)):
                comments_to_predict[i][pred_label_type] = float(predictions[i])
                comments_to_predict[i]['upvote'] = float(predictions[i]) * label_std + label_mean if "z" in args.label_type else np.exp(float(predictions[i]))
                out_dict[comments_to_predict[i]['idx']] = {'comment': comments_to_predict[i]['comment'], 
                                                           pred_label_type: comments_to_predict[i][pred_label_type], 
                                                           'upvote': comments_to_predict[i]['upvote']}
            # append to output file
            with open(out_filepath, "a+") as f:
                json.dump(out_dict, f)
                f.write('\n')
            # append the prediction for the real comments to predictions_only.csv
            with open(os.path.join(new_model_dir, f"predictions_only.csv"), "a+") as file:
                file.write(f"{sample['id']},{sample['idx']},{predictions[0]},{orig_label}\n")
        
        if global_i % 100 == 0: 
            print(f"ID={sample['id']}: {predictions[0]} ({orig_label}), {predictions[1]}, {predictions[2]}, {predictions[3]}, {predictions[4]}, {predictions[5]}")


def model_predict(model, tokenizer, examples, batch_size):
    predictions_all = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:min(i+batch_size, len(examples))]
        tokenized = reformat_comments(batch, tokenizer)
        predictions = model(**tokenized).logits[:,0].tolist()
        predictions_all.extend(predictions)
    return predictions_all


def reformat_comments(examples, tokenizer):
    input_texts = []
    for d in examples:
        comment = d['comment']
        title = d['submission_title']
        reply_time = d['created_comment']
        post_time = d['created_submission']
        author = d['author']
        author_id = d['author_id']

        input_text = f"{title} || {comment}"
        if args.time == "unix": input_text = f"Time of reply: {reply_time}, which is {reply_time-post_time} after post. || {input_text}"
        elif args.time == "string_old": input_text = f"{input_text} || Reply time after post: {reply_time-post_time}. "
        elif args.time == "readable": input_text = f"Time of reply: {datetime.fromtimestamp(reply_time).strftime('%A %B %d %Y %H:%M:%S %p')}, which is {seconds_to_time_string(reply_time-post_time)} after post. || {input_text}"
        if args.author != None: input_text = f"Author: {author} ({author_id}) || {input_text}"
        input_texts.append(input_text)
        
    tokenized = tokenizer(input_texts, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="pt").to(device)
    return tokenized

if args.train: main_train_upvote_model()
if args.filter: main_filter_original_comments()
if args.predict: main_predict_synthetic_comments()