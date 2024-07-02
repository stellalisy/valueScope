import sys
import os
import json
import random
import pandas as pd

filter_phrases = ['I apologize, but', 'not able to fulfill this request', 'cannot fulfill your request', "I cannot provide a rewritten", "rewritten comment"]
split_phrases = ['"\n\nThis', '"\n\nRating:', '"\n\nPlease', '"\n\nNote', '"\n\nRewritten', '"\n\nOriginal', '"\n\nRemember', '"\n\nI rewrote', '"\n\nI hope', '"\n\nI rated', '"\n\nI Changed', '"\n\nI kept', '"\n\nI tried', '"\n\nThe original', '"\n\nI\'ve']

def filter_comment(comment):
    if len(comment) == 0:
        return None
    if any([phrase in comment for phrase in filter_phrases]):
        return None
    for split_phrase in split_phrases:
        if split_phrase in comment:
            return comment.split(split_phrase)[0]
    return comment

def process_data(df, subredditname=None):
    _data = []
    for _, row in df.iterrows():
        original_id = row['id']
        _data.append({'subreddit': subredditname, 'id': original_id + "-0", 'id-original': original_id, 'comment': row['original_comment'], 'rating': row['original_rating'], 'post_title': row['submission_title']})
        for scale in ['1', '2', '3', '4', '5']:
            filtered_comment = filter_comment(row[scale])
            if filtered_comment is not None:
                _data.append({'subreddit': subredditname, 'id': original_id + "-" + scale, 'id-original': original_id, 'comment': filtered_comment, 'rating': int(scale), 'post_title': row['submission_title']})
    return _data

# Function to determine if IDs need to be reversed
def bool_reversed_id(id_pair):
    id_pair = id_pair.replace("t1_", "")
    id_pair = id_pair.replace("t3_", "")
    id_pair = id_pair.replace("t2_", "")
    id1, id2 = id_pair.split("_")
    return True if id1 > id2 else False

# Function to sort ID pair
def sort_id_pair(id_pair):
    id_pair = id_pair.replace("t1_", "")
    id_pair = id_pair.replace("t3_", "")
    id_pair = id_pair.replace("t2_", "")
    if bool_reversed_id(id_pair):
        id1, id2 = id_pair.split("_")
        return id2 + "_" + id1
    else:
        return id_pair

# Define target category from command-line argument or default
TARGET_CAT = sys.argv[1] if len(sys.argv) > 1 else "politics"
assert TARGET_CAT in ['politics', 'gender', 'science', 'finance']

# Define subreddit and dimension mappings
SUBREDDITS = {'politics': set(['democrats', 'republican', 'libertarian']), 'gender': set(['askmen', 'askwomen', 'asktransgender']), 'science': set(['askscience', 'shittyaskscience', 'asksciencediscussion']), 'finance': set(['pennystocks', 'stocks', 'wallstreetbets', 'wallstreetbetsnew'])}
TARGET_SUBREDDITS = SUBREDDITS[TARGET_CAT]
TARGET_DIMENSIONS = ["humorous", "formality", "politeness", "supportiveness", "sarcasm"]

print("start processing...")
data_all, id_win_rate_all = {}, {}

# Iterate over target dimensions
for TARGET_DIMENSION in TARGET_DIMENSIONS:
    print("*" * 30)
    print(" " * 14 + TARGET_DIMENSION)
    print("*" * 30)

    # Define paths for input and output files
    PATH_RES = f"<your_path_here>/{TARGET_CAT}-{TARGET_DIMENSION}.jsonl"
    PATH_OUT = f"<your_path_here>/winrate-{TARGET_CAT}-{TARGET_DIMENSION}.jsonl"

    data = []

    # Iterate over files in comments directory
    for filename in os.listdir(PATH_COMMENTS):
        if not TARGET_DIMENSION[:5] in filename:
            continue
        if "_old" in filename:
            continue
        subreddit = filename.split("_")[0]
        if subreddit in TARGET_SUBREDDITS:
            df_file = pd.read_json(PATH_COMMENTS + filename, lines=True)
            processed = process_data(df_file, subredditname=subreddit)
            print(filename, len(df_file), len(processed))
            data.extend(processed)

    data = pd.DataFrame(data)
    data = data.drop_duplicates(subset=['id'])
    data['id'] = data['id'].str.replace("t1_", "")
    data['id-original'] = data['id-original'].str.replace("t1_", "")
    print("total: ", len(data))

    if len(data) == 0:
        continue
    if not os.path.isfile(PATH_RES):
        continue

    # Load binary prediction results
    df = pd.read_json(PATH_RES, lines=True)
    print("# loaded pairs: ", len(df))

    df['id_pair'] = df['id_pair'].str.replace("t1_", "")
    df['id_pair_sorted'] = df['id_pair'].apply(sort_id_pair)
    df['reversed'] = df['id_pair'].apply(bool_reversed_id)
    map_reverse = {1: 0, 0: 1}
    df['label'] = df.apply(lambda row: map_reverse[row['label']] if row['reversed'] else row['label'], axis=1)

    df = df[['id_pair_sorted', 'label']].drop_duplicates(subset=['id_pair_sorted'])
    df['id1'] = df['id_pair_sorted'].apply(lambda x: x.split("_")[0])
    df['id2'] = df['id_pair_sorted'].apply(lambda x: x.split("_")[1])
    print("after dedupling: ", len(df))

    num_pairs = len(data) * (len(data) - 1) / 2
    print(f"pair coverage: {len(df) / num_pairs * 100:.2f}% ({len(df) / 1000000:.0f}M/{num_pairs / 1000000:.0f}M)")

    df2 = df
    id_win = {}
    total_win = {0: 0, 1: 0}

    for (comment_id, label), _df in df2.groupby(["id1", "label"]):
        if comment_id not in id_win:
            id_win[comment_id] = {0: 0, 1: 0}
        id_win[comment_id][label] += len(_df)
        total_win[label] += len(_df)

    label_reverse = {0: 1, 1: 0}

    for (comment_id, label), _df in df2.groupby(["id2", "label"]):
        if comment_id not in id_win:
            id_win[comment_id] = {0: 0, 1: 0}
        id_win[comment_id][label_reverse[label]] += len(_df)
        total_win[label_reverse[label]] += len(_df)

    print(f"# of comments: {len(id_win)} (coverage: {len(id_win) / len(data) * 100:.1f}%)")

    id_num_label = {k: sum(v.values()) for k, v in id_win.items()}
    id_num_label = pd.DataFrame(id_num_label.items(), columns=['id', 'num_label'])
    print(f"quantiles: {id_num_label.num_label.quantile(0.25)}/{id_num_label.num_label.quantile(0.5)}/{id_num_label.num_label.quantile(0.75)}")
    print(f"avg # of labels per comment: {id_num_label.num_label.mean():.1f}")

    id_win_rate = {}

    for comment_id, win_count in id_win.items():
        id_win_rate[comment_id] = win_count[0] / (win_count[1] + win_count[0])

    print(f"average win rate: {sum(list(id_win_rate.values())) / len(id_win_rate)}")
    print("total # of win: ", total_win)

    data['winrate'] = data['id'].apply(lambda x: id_win_rate.get(x, None))
    data['num_win'] = data['id'].apply(lambda x: id_win[x][0] if x in id_win else None)
    data['num_lost'] = data['id'].apply(lambda x: id_win[x][1] if x in id_win else None)
    avg_original, avg_generated = data[data.id.str.contains("-0")].winrate.mean(), data[~data.id.str.contains("-0")].winrate.mean()

    data.to_json(PATH_OUT, orient='records', lines=True)

    print(f"Average winrate of original/generated comments: {avg_original:.3f}/{avg_generated:.3f}")
    print(data[data.id.str.contains("-0")].groupby(['rating']).mean(['winrate']))
    print(data[~data.id.str.contains("-0")].groupby(['rating']).mean(['winrate']))
