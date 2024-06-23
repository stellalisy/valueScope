import os
import json
import logging
import sys
import time
import pickle
from collections import defaultdict
from transformers import BitsAndBytesConfig, pipeline
import torch
import argparse
import warnings
warnings.filterwarnings('ignore')

import prompts

parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does', epilog='Text at the bottom of help')
parser.add_argument('-r', '--subreddit', type=str, required=True)
parser.add_argument('-n', '--norm_dimension', type=str, required=True)
parser.add_argument('-d', '--data_dir', type=str, default="/gscratch/argon/stelli/reddit_norm/upvote_prediction/data/processed_upvotes/")
parser.add_argument('-o', '--output_dir', type=str, default="/gscratch/argon/stelli/reddit_norm/style_transfer/data/output/llama3/")
parser.add_argument('-m', '--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument('-u', '--upvote_dir', type=str, default=None)
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('-sb', '--sub_batch_size', type=int, default=6)
parser.add_argument('-q', '--quantization', type=int, default=None)
parser.add_argument('-s', '--start_idx', type=int, default=0)
parser.add_argument('-e', '--end_idx', type=int, default=100000000)
args = parser.parse_args()

default_model_dir = {
    "askmen": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/askmen_dialogueRPT_log_half_2ep_time",
    "askwomen": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/askwomen_dialogueRPT_log_half_2ep_time",
    "asktransgender": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/asktransgender_dialogueRPT_log_half_5ep_time",
    "askscience": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/askscience_dialogueRPT_log_half_5ep_time",
    "asksciencediscussion": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/asksciencediscussion_dialogueRPT_log_half_5ep_time",
    "shittyaskscience": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/shittyaskscience_dialogueRPT_log_half_5ep_time",
    "democrats": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/democrats_dialogueRPT_log_half_5ep_time",
    "republican": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/republican_dialogueRPT_log_half_5ep_time",
    "libertarian": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/libertarian_dialogueRPT_log_half_2ep_time",
    "wallstreetbets": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/wallstreetbets2m_dialogueRPT_log_2m_2ep_time",
    "stocks": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/stocks_dialogueRPT_log_half_5ep_time",
    "pennystocks": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/pennystocks_dialogueRPT_log_half_5ep_time",
    "wallstreetbetsnew": "/gscratch/argon/stelli/reddit_norm/upvote_prediction/upvote_prediction_models/wallstreetbetsnew_dialogueRPT_log_half_5ep_time"
}
if args.upvote_dir == None and args.subreddit in default_model_dir:
    args.upvote_dir = default_model_dir[args.subreddit]


logging.basicConfig(filename=f'/gscratch/argon/stelli/reddit_norm/style_transfer/logs/llama3/log/style_transfer_llama3_{args.subreddit}_{args.norm_dimension}.log', 
                    format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', encoding='utf-8', level=logging.DEBUG)
for k, v in vars(args).items():
    logging.info(f"{k}: {v}")

def load_llama3(model_name="meta-llama/Meta-Llama-3-8B-Instruct", model_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.quantization == 4:
        bnb_config_4b = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, "float16")
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config_4b,
        )
    elif args.quantization == 8:
        bnb_config_8b = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config_8b,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return model, tokenizer, pipe

def get_response_llama3(messages, temperature=1, max_tokens=256, top_p=1, frequency_penalty=0, presence_penalty=0):
    """
    messages: list of messages (works for batching)
    """
    if type(messages[0]) == dict:
        # batch size is 1
        messages = [messages]

    tokenizer.pad_token = tokenizer.eos_token  # add padding token
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    # input_ids = tokenizer.apply_chat_template(messages, padding=True, max_length=4096, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # outputs = model.generate(input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature, top_p=top_p, return_dict_in_generate=True, output_scores=True) # for getting logprobs
    # sequence = outputs.sequences[0][input_ids.shape[-1]:]
    # response_text = tokenizer.decode(sequence, skip_special_tokens=True)

    inputs = [tokenizer.apply_chat_template(message,tokenize=False) for message in messages]
    input_ids = tokenizer(inputs, padding='longest', add_special_tokens=True, return_tensors="pt").to(model.device)
    outputs = model.generate(**input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, temperature=temperature, top_p=top_p) #, return_dict_in_generate=True, output_scores=True) # for getting logprobs

    
    output_ids = outputs[:,input_ids.input_ids.shape[-1]:]
    response_texts = [text.replace("assistant\n\n", "") for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
    return response_texts


def comment_to_submission(subreddit):
    lookup_dir = "/gscratch/argon/stelli/reddit_norm/style_transfer/data/id_to_title/"
    if not os.path.exists(os.path.join(lookup_dir, f"{subreddit}_id_to_title.pkl")):
        logging.info(f"Creating lookup table for r/{subreddit}")
        with open(os.path.join("/gscratch/argon/stelli/reddit_norm/style_transfer/data/dump_data", f"submissions_{subreddit}.pkl"), "rb") as f:
            data_submissions = pickle.load(f)
        id_to_title = {submission["id"]: submission["title"] for submission in data_submissions}
        with open(os.path.join(lookup_dir, f"{subreddit}_id_to_title.pkl"), "wb") as f:
            pickle.dump(id_to_title, f)
    else:
        with open(os.path.join(lookup_dir, f"{subreddit}_id_to_title.pkl"), "rb") as f:
            id_to_title = pickle.load(f)
    return id_to_title

def nested_dict(): 
    return defaultdict(nested_dict)
def comment_to_id(subreddit):
    lookup_dir = "/gscratch/argon/stelli/reddit_norm/style_transfer/data/comment_to_id/"
    if not os.path.exists(os.path.join(lookup_dir, f"{subreddit}_time_to_author_to_score_to_comment_to_id.pkl")):
        logging.info(f"Creating lookup table for r/{subreddit}")
        with open(os.path.join("/gscratch/argon/hjung10/norm_discovery_project/data/data_dump", f"comments_{subreddit}.pkl"), "rb") as f:
            dump_data = pickle.load(f)
        time_to_author_to_score_to_comment_to_id = defaultdict(nested_dict)
        for comment in dump_data:
            time_to_author_to_score_to_comment_to_id[comment['created_utc']][comment['author']][comment['score']][comment['body']] = comment['id']
        with open(os.path.join(lookup_dir, f"{subreddit}_time_to_author_to_score_to_comment_to_id.pkl"), "wb") as f:
            pickle.dump(time_to_author_to_score_to_comment_to_id, f)
    else:
        with open(os.path.join(lookup_dir, f"{subreddit}_time_to_author_to_score_to_comment_to_id.pkl"), "rb") as f:
            time_to_author_to_score_to_comment_to_id = pickle.load(f)
    return time_to_author_to_score_to_comment_to_id

def transform_comment(comment, norm_dimension):
    comment_transformed = {
        "id": comment["id"],
        "original_comment": comment["body"] if "body" in comment else comment["comment"],
        "original_rating": None,
        "original_score": comment["score"],
        "submission_title": comment["submission_title"] if "submission_title" in comment else None,
    }
    
    messages = [get_rewrite_prompt(comment, norm_dimension, rating) for rating in range(5)]
    messages.append(rate_original_prompt(comment, norm_dimension))

    response_texts = get_response_llama3(messages)
    for rating in range(5):
        comment_transformed[rating+1] = response_texts[rating].replace("REWRITTEN COMMENT:", "").strip()

    orig_rating = get_response_rating(response_texts[-1], norm_dimension)
    comment_transformed["original_rating"] = orig_rating

    return comment_transformed


def transform_comment_batch(batch, norm_dimension):
    comment_transformed_batch = [{
        "id": comment["id"],
        "idx": comment["idx"],
        "original_comment": comment["body"] if "body" in comment else comment["comment"],
        "original_rating": None,
        "original_score": comment["label"],
        "submission_title": comment["submission_title"] if "submission_title" in comment else None,
    } for comment in batch]
    
    messages = []
    for comment in batch:
        messages.extend([get_rewrite_prompt(comment, norm_dimension, rating) for rating in range(5)])
        messages.append(rate_original_prompt(comment, norm_dimension))

    response_texts_batch = []
    for i in range(0, len(messages), args.sub_batch_size):
        response_texts_batch.extend(get_response_llama3(messages[i:i+args.sub_batch_size]))
    # response_texts_batch = get_response_llama3(messages)
    for i, comment in enumerate(batch):
        response_texts = response_texts_batch[i*6:(i+1)*6]
        for rating in range(5):
            comment_transformed_batch[i][rating+1] = response_texts[rating].replace("REWRITTEN COMMENT:", "").strip()

        orig_rating = get_response_rating(response_texts[-1], norm_dimension)
        comment_transformed_batch[i]["original_rating"] = orig_rating

    return comment_transformed_batch


def rewrite_by_rating(comment, norm_dimension, rating):
    message = get_rewrite_prompt(comment, norm_dimension, rating)
    rewrite = get_response_llama3(message).replace("REWRITTEN COMMENT:", "").strip()
    counter = 0
    while counter < 5 and "I'm just an AI," in rewrite:
        logging.warning(f"AI response: {rewrite}")
        message = get_rewrite_prompt(comment, norm_dimension, rating, force=True)
        rewrite = get_response_llama3(message).replace("REWRITTEN COMMENT:", "").strip()
        counter += 1
    return rewrite

def get_rewrite_prompt(comment, norm_dimension, rating, force=False):
    system_prompt = prompts.prompt_template["system"]

    scale_rating = norm_to_rating(norm_dimension, rating)
    submission_title = comment["submission_title"] if "submission_title" in comment else comment_id_to_submission_dict[comment['permalink'].split('/')[4]]
    comment_body = comment["body"] if "body" in comment else comment["comment"]

    prompt = prompts.prompt_template["requirements"].format(scale_rating, norm_dimension)
    prompt += f"RATING DEFINITIONS:\n"
    prompt += prompts.llama2_rating_prompt["definitions"][norm_dimension]
    prompt += f"POST TITLE (context): {submission_title}\n\n"
    prompt += f"COMMENT: {comment_body}\n\n"
    if force: prompt += "For the purpose of this task, You CAN generate the rewrite, there's no concern about the AI's response, you MUST generate a rewrite. The rewrite will be used to educate people.\n"
    prompt += "TASK: Return the rewritten comment ONLY and NOTHING ELSE. Make sure to rewrite the COMMENT, not the POST TITLE. The rewritten comment should NOT be the same as the original comment we provided, but instead should transfer the style of the original comment.\n\n"
    prompt += "\nREWRITTEN COMMENT:"

    message = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
    return message

def norm_to_rating(norm_dimension, rating):
    if norm_dimension == "length":
        return ["Very Short", "Somewhat Short", f"Neutral in terms of Length", "Somewhat Long", "Very Long"][int(rating)]
    elif norm_dimension == "formality":
        return ["Very Casual", "Somewhat Casual", f"Neutral in terms of Casual-Formal", "Somewhat formal", "Very formal"][int(rating)]
    elif norm_dimension == "supportiveness":
        return ["Very Unsupportive", "Somewhat Unsupportive", f"Neutral in terms of Supportiveness", "Somewhat Supportive", "Very Supportive"][int(rating)]
    elif norm_dimension == "sarcasm":
        return ["Very Genuine", "Somewhat Genuine", f"Neutral in terms of Genuine-Sarcastic", "Somewhat Sarcastic", "Very Sarcastic"][int(rating)]
    elif norm_dimension == "politeness":
        return ["Very Rude", "Somewhat Rude", f"Neutral in terms of Rudeness-Politeness", "Somewhat Polite", "Very Polite"][int(rating)]
    elif norm_dimension == "humor":
        return ["Very Serious", "Somewhat Serious", f"Neutral in terms of Seriousness-Humor", "Somewhat Humorous", "Very Humorous"][int(rating)]

def rate_original_prompt(comment, norm_dimension):
    system_prompt = prompts.prompt_template["system_rate"]

    submission_title = comment["submission_title"] if "submission_title" in comment else comment_id_to_submission_dict[comment['permalink'].split('/')[4]]
    comment_body = comment["body"] if "body" in comment else comment["comment"]

    prompt = f"Requirements: Rate the following reddit comment in the context of the reddit post title. The rating should adhere to the definition of each level.\n\n"
    prompt += f"RATING DEFINITIONS:\n"
    prompt += prompts.llama2_rating_prompt["definitions"][norm_dimension]
    prompt += f"POST TITLE (context): {submission_title}\n\n"
    prompt += f"COMMENT: {comment_body}\n\n"
    prompt += "TASK: Return the rating ONLY and NOTHING ELSE.\n\n"
    prompt += "RATING:"

    message = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
    # rating = get_response_rating(message, norm_dimension)
    return message

def get_response_rating(response_text, norm_dimension):
    rating = response_text.strip().replace("RATING:", "").strip()
    for r in range(5):
        if r != 2 and norm_to_rating(norm_dimension, r).lower() in rating.lower():
            return r+1
        if r == 2 and ("in-between" in rating.lower() or "neutral" in rating.lower()):
            return r+1
    if "NO RATING" in rating:
        return 3  # in-between/neutral
    if norm_to_rating(norm_dimension, 1).split(" ")[-1] in rating.lower():
        return 1
    if norm_to_rating(norm_dimension, 3).split(" ")[-1] in rating.lower():
        return 3
    if rating.replace('.','').strip().isdigit():
        temp_rating = int(rating.replace('.','').strip())
        if temp_rating > 0 and temp_rating < 6:
            return temp_rating
    logging.warning(f"Rating not found in response (returning rating=3): {response_text}")
    return 3


def write_output(comment_transformed, subreddit, norm_dimension):
    if type(comment_transformed) == list:
        for comment in comment_transformed:
            with open(os.path.join(args.output_dir, f"{subreddit}_{norm_dimension}.jsonl"), "a+") as f:
                json.dump(comment, f)
                f.write("\n")
    else:
        with open(os.path.join(args.output_dir, f"{subreddit}_{norm_dimension}.jsonl"), "a+") as f:
            json.dump(comment_transformed, f)
            f.write("\n")


def transform_comments(target_subreddit, norm_dimension):

    # load original comments data
    if args.subreddit == 'wallstreetbets' and os.path.exists('/gscratch/argon/stelli/reddit_norm/upvote_prediction/data/processed_upvotes/wallstreetbets_inference_1m.pkl'):
        with open('/gscratch/argon/stelli/reddit_norm/upvote_prediction/data/processed_upvotes/wallstreetbets_inference_1m.pkl', "rb") as f:
            comments_all = pickle.load(f)
    else:
        with open(os.path.join(args.data_dir, f"{target_subreddit}.pkl"), "rb") as f:
            comments_all = pickle.load(f)
    total_num_comments = len(comments_all)
    # logging.info(f"Total comments in subreddit r/{target_subreddit}: {total_num_comments}")
    
    # using the inference half of the upvote prediction data
    with open(os.path.join(args.upvote_dir, f"inference_sample_ids.txt"), "r") as f:
        filtered_ids = [int(line.strip()) for line in f.readlines()]
    # logging.info(f"Filtered IDs: {len(filtered_ids)}")
    if args.subreddit == 'wallstreetbets': filtered_ids = filtered_ids[:1000000]


    with open(os.path.join("/gscratch/argon/stelli/reddit_norm/style_transfer/data/idx_to_id", f"{target_subreddit}_idx_to_id.pkl"), "rb") as f:
        idx_to_id = pickle.load(f)

    comments = []
    if args.subreddit == 'wallstreetbets' and os.path.exists('/gscratch/argon/stelli/reddit_norm/upvote_prediction/data/processed_upvotes/wallstreetbets_inference_1m.pkl'):
        comments_all = {comment['idx']: comment for comment in comments_all}

    for i in filtered_ids:
        if type(comments_all) == list and i >= len(comments_all):
            logging.warning(f"Index {i} out of range, skipping -- THIS SHOULD NOT HAPPEN")
            continue
        if type(comments_all) == dict and i not in comments_all:
            logging.warning(f"Index {i} not found in comments, skipping -- THIS SHOULD NOT HAPPEN")
            continue
        temp_comment = comments_all[i]
        temp_comment["idx"] = i
        if "id" in temp_comment: 
            comments.append(temp_comment)
            continue
        try: temp_id = idx_to_id[i]
        except: temp_id = None 
        temp_comment["id"] = temp_id
        comments.append(temp_comment)
        # if temp_comment['created_comment'] > 1640998800: print(f"{i}: {temp_comment['created_comment']}")
    logging.info(f"Total comments in subreddit r/{args.subreddit}: {total_num_comments}, filtered (kept) comments: {len(comments)}")
    print(f"Total comments in subreddit r/{args.subreddit}: {total_num_comments}, filtered (kept) comments: {len(comments)}")

    if args.subreddit == 'wallstreetbets' and not os.path.exists('/gscratch/argon/stelli/reddit_norm/upvote_prediction/data/processed_upvotes/wallstreetbets_inference_1m.pkl'):
        with open('/gscratch/argon/stelli/reddit_norm/upvote_prediction/data/processed_upvotes/wallstreetbets_inference_1m.pkl', "wb") as f:
            pickle.dump(comments, f)
        print(f"Saved inference {len(comments)} comments to /gscratch/argon/stelli/reddit_norm/upvote_prediction/data/processed_upvotes/wallstreetbets_inference_1m.pkl")
                             
    try: 
        comments = comments[args.start_idx:min(args.end_idx, len(comments))]
        logging.info(f"Processing comments from index {args.start_idx} to {args.end_idx}")
    except: 
        pass
        logging.info(f"Start & end range not specified / invalid, processing all remaining comments")
    


    output_filepath = os.path.join(args.output_dir, f"{target_subreddit}_{norm_dimension}.jsonl")
    if not os.path.exists(output_filepath): 
        comments_processed = []
    else:
        with open(output_filepath, "r") as f:
            comments_processed = [json.loads(line) for line in f.readlines()]
        logging.info(f"Number of processed comments: {len(comments_processed)}")
        
        # sanity check: all processed comments should be in the inference filter
        comments_processed = [comment for comment in comments_processed if comment["idx"] in filtered_ids]
        num_comments_before_dedup = len(comments_processed)
        logging.info(f"Processed comments that are in the inference filter: {num_comments_before_dedup} (should be equal to the number of processed comments)")
        
        # deduplicate comments_processed by idx
        comments_processed = list({comment["idx"]: comment for comment in comments_processed}.values())
        num_comments_after_dedup = len(comments_processed)
        logging.info(f"Deduplicated processed comments: {num_comments_after_dedup} (removed {num_comments_before_dedup - num_comments_after_dedup} duplicated comments)")
        
        if num_comments_before_dedup != num_comments_after_dedup:

            logging.warning(f"Deduplication removed {num_comments_before_dedup - num_comments_after_dedup} comments, saving deduplicated comments to file: {output_filepath}")
            with open(output_filepath, 'w') as f:
                for comment in comments_processed:
                    json.dump(comment, f)
                    f.write("\n")

    processed_idxs = [comment["idx"] for comment in comments_processed]

    num_comments_in_range = len(comments)
    comments = [comment for comment in comments if comment["idx"] not in processed_idxs]
    num_unprocessed_in_range = len(comments)
    num_processed_in_range = num_comments_in_range - num_unprocessed_in_range
    logging.info(f"Processed comments in specified range [{args.start_idx}, {args.end_idx}): {num_processed_in_range}, remaining comments in specified range: {num_unprocessed_in_range}")
    if len(comments) == 0: 
        print(f"No comments left to process for r/{target_subreddit}!!!! YOU ARE DONE")
        logging.info(f"No comments left to process for r/{target_subreddit}!!!! YOU ARE DONE")

    # process comments
    i = 0
    while i < len(comments):
        
        sys.stdout.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] r/{target_subreddit}: {i}/{len(comments)}\n')
        logging.info(f"r/{target_subreddit}-{norm_dimension}: {i}/{len(comments)} | id={comments[i]['id']} | idx={comments[i]['idx']}")

        batch = comments[i:min(i+args.batch_size, len(comments))]
        # comment = comments[i]

        comment_transformed_batch = transform_comment_batch(batch, norm_dimension)
        write_output(comment_transformed_batch, target_subreddit, norm_dimension)
        i += args.batch_size
    print(f"Finished processing comments for r/{target_subreddit}!!!! YOU ARE DONE")
    logging.info(f"Finished processing comments for r/{target_subreddit}!!!! YOU ARE DONE")

if args.subreddit == "askmen" or args.subreddit == "askwomen": 
    comment_id_to_submission_dict = comment_to_submission(args.subreddit)
# time_to_author_to_score_to_comment_to_id = comment_to_id(args.subreddit)
model, tokenizer, pipe = load_llama3()
transform_comments(args.subreddit, args.norm_dimension)
