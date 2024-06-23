import os
import json
import pickle
from collections import defaultdict
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subreddit", type=str, default="askmen")
    return parser.parse_args()
args = get_args()

subreddit = args.subreddit
data_dir = "/gscratch/argon/stelli/reddit_norm/upvote_prediction/processed_upvotes"
data_filename = os.path.join(data_dir, f"{subreddit}.pkl")

output_dir = "/gscratch/argon/stelli/reddit_norm/style_transfer/data/idx_to_id"
if os.path.exists(os.path.join(output_dir, f"{subreddit}_idx_to_id.pkl")): 
    print(f"idx_to_id for {subreddit} already exists: {os.path.join(output_dir, f'{subreddit}_idx_to_id.pkl')}")
    exit()

with open(data_filename, "rb")as f:
    comments_all = pickle.load(f)

total_num_comments = len(comments_all)
print(f"total comments in {subreddit}: {total_num_comments}")

def nested_dict(): 
    return defaultdict(nested_dict)
def comment_to_id(subreddit):
    lookup_dir = "/gscratch/argon/stelli/reddit_norm/style_transfer/data/comment_to_id/"
    if not os.path.exists(os.path.join(lookup_dir, f"{subreddit}_time_to_author_to_score_to_comment_to_id.pkl")):
        print(f"Creating lookup table for r/{subreddit}")
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

time_to_author_to_score_to_comment_to_id = comment_to_id(subreddit)

idx_to_id = {}
for idx, comment in enumerate(comments_all):
    if 'id' in comment:
        comment_id = comment['id']
    else:
        comment_id = time_to_author_to_score_to_comment_to_id[comment["created_comment"]][comment["author"]][comment["label"]][comment["comment"]]
    idx_to_id[idx] = comment_id
    if idx % 100000 == 0: print(f"done processing idx={idx}")

print("first id:", idx_to_id[0])
with open(os.path.join(output_dir, f"{subreddit}_idx_to_id.pkl"), "wb") as f:
    pickle.dump(idx_to_id, f)