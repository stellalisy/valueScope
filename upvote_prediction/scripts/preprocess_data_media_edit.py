import zstandard
import os
import json
import pickle
import sys
from datasets import load_from_disk
from datetime import datetime
import logging.handlers
import argparse

log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subreddit", type=str, default="askmen")
    parser.add_argument("--no_cache", action="store_true")
    return parser.parse_args()
args = get_args()


def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)

            if not chunk:
                break
            lines = (buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line, file_handle.tell()

            buffer = lines[-1]

        reader.close()

def process_line_comment(line):
    bool_badline = False
    try:
        obj = json.loads(line)
        # breakpoint()
        if not obj['subreddit'].lower() in subset_subreddits:
            bool_badline = True
        elif obj['body'] in FILTERED_COMMENTS:
            bool_badline = True
        link_or_parent_id = str(obj['link_id']) if 'link_id' in obj else str(obj['parent_id'])
    except (KeyError, json.JSONDecodeError) as err:
        bool_badline = True
    if link_or_parent_id.startswith("t1_"):
        # ignore second level comments
        return True, obj
    try:
        new_obj = obj
        new_obj.update({'created_utc':obj['created_utc'], 'parent_id': link_or_parent_id, 'body': obj['body'], 'score': obj['score'], 'name': obj['name'], 'author': obj['author']})
    except:
        new_obj = obj
        new_obj.update({'created_utc':obj['created_utc'], 'parent_id': link_or_parent_id, 'body': obj['body'], 'score': obj['score'], 'name': "t1_"+obj['id'], 'author': obj['author']})
    assert link_or_parent_id.startswith("t3_"), str(obj)+"\n\n"+str(new_obj)
    assert new_obj['name'].startswith("t1_"), str(obj)+"\n\n"+str(new_obj)
    return bool_badline, new_obj

def process_line_submission(line):
    bool_badline = False
    try:
        obj = json.loads(line)
        # breakpoint()
        # created = datetime.utcfromtimestamp(int(obj['created_utc']))
        if obj["author"] == "[deleted]":
            bool_badline = True
        elif obj["selftext"] in FILTERED_SUBMISSIONS:
            bool_badline = True
        elif obj['removed_by_category'] != None or not obj['subreddit'].lower() in subset_subreddits:
            bool_badline = True
    except (KeyError, json.JSONDecodeError) as err:
        bool_badline = True
    return bool_badline, obj


if __name__ == "__main__":
    # CONFIG
    # base_path = "/gscratch/argon/chanyoun/reddit_dumps/bysub/"
    base_path = "/gscratch/argon/hjung10/norm_discovery_project/data/data_reddit_dump/" # hayoung's data dir
    path_out = "/gscratch/argon/stelli/reddit_norm/upvote_prediction/data/processed_upvotes_filter/"
    
    # subset_subreddits = set(["askmen", "askwomen", "rateme", "amiugly", "amiuglybrutallyhonest", "explainlikeimfive", "askphysics", "moderatepolitics", "truerateme", "politics", "neutralpolitics"])
    # subset_subreddits = set(["askmen", "askwomen"])
    # subset_subreddits = set(["stocks"]) #"stocks", "wallstreetbets", "pennystocks", "asktransgender"
    subset_subreddits = set([args.subreddit])
    MAX_TIME_DIFF = 86400 # ignore comments posted after this much of time as they may not get upvotes as they normally would
    path_processed_hp_data = "/home/chanyoun/datasets/chp"

    FILTERED_COMMENTS = set(["[removed]", "[deleted]"])
    FILTERED_SUBMISSIONS = set(["[removed]", "[deleted]"])
    for subreddit in subset_subreddits:
        path_to_save = path_out+subreddit+".pkl"
        if os.path.exists(path_to_save) and not args.no_cache:
            print(f"{path_to_save} already exists. Continue...")
            continue
        # if os.path.isfile(path_out+subreddit+".pkl"):
        #     print(path_out+subreddit+".pkl", "already exists. Continue...")
        #     continue
        print("start processing", subreddit)
        data = {"comments":[], "submissions":[]}
        for file_path in os.listdir(base_path):
            if not file_path.split("_")[0].lower() == subreddit:
                continue
            subreddit_real_name = file_path.split("_")[0].lower()
            print(subreddit_real_name)
            
            file_type = "comments" if "_comments.zst" in file_path else "submissions"
            file_size = os.stat(base_path+file_path).st_size
            file_lines = 0
            file_bytes_processed = 0
            created = None
            bad_lines = 0
            print(f"{file_path} ({file_type}, {file_size})")
            for line, file_bytes_processed in read_lines_zst(base_path+file_path):
                if file_type == "comments":
                    # print("comments")
                    bool_badline, obj = process_line_comment(line)
                elif file_type == "submissions":
                    # print("submissions")
                    bool_badline, obj = process_line_submission(line)
                if bool_badline:
                    bad_lines += 1
                else:
                    data[file_type].append(obj)
                file_lines += 1
                if file_lines % 100000 == 0:
                    log.info(f"{file_lines:,} : {bad_lines:,} : {file_lines-bad_lines:,} : {(bad_lines/file_lines*100):.1f}%:{(file_bytes_processed / file_size) * 100:.0f}%")
            log.info(f"Completed {file_path}: {file_lines:,} : {bad_lines:,}")
        
        submission_ids = {d['id']: d for d in data['submissions']}
        comments_filtered = [d for d in data["comments"] if d['parent_id'].split("_")[1] in submission_ids]
        data["comments"] = comments_filtered

        hp_train_data = [] # [(d['c_root_id_A'], d['c_root_id_B']) for d in load_from_disk(path_processed_hp_data)['train']]
        hp_train_data = set([cid for t in hp_train_data for cid in t])
        
        data_to_save = []
        ignored_due_to_time_diff = 0
        for comment in comments_filtered:
            if comment['name'] in hp_train_data:
                # filter comments that are in the human preference data (training set)
                print("skipping a comment in the hp data")
                continue
            submission = submission_ids[comment['parent_id'].split("_")[1]]
            created_comment = comment['created_utc']
            created_submission = submission['created_utc']
            if created_comment - created_submission > MAX_TIME_DIFF: 
                # filter comments that are not posted within MAX_TIME_DIFF
                ignored_due_to_time_diff += 1
                continue
            d = {}
            d['id'] = comment['name']
            d['submission_title'] = submission['title']
            d['submission_body'] = submission['selftext']
            d['comment'] = comment['body']
            d['created_comment'] = created_comment
            d['created_submission'] = created_submission
            d['label'] = int(comment['score'])
            d['author'] = comment['author']
            d['author_id'] = comment['name']
            d['submission_has_media'] = (submission['media'] != None or submission['is_video'])
            d['id_from_dump'] = comment['id']
            d['retrieved_on'] = comment['retrieved_on'] if 'retrieved_on' in comment else None
            d['edited'] = comment['edited'] if 'edited' in comment else False
            data_to_save.append(d)
        
        edit_filer, link_filter, retrieve_time_filter, media_filter = [], [], [], []
        not_filtered = []
        for i, sample in enumerate(data_to_save):
            filtered = False
            if 'edited' in sample and sample['edited']:
                edit_filer.append(sample['id'])
                filtered = True
            if len(sample['comment'].split()) == 1 and 'http' in sample['comment']:
                link_filter.append(sample['id'])
                filtered = True
            if sample['retrieved_on'] != None and (sample['retrieved_on'] - sample['created_comment']) < 86400:
                retrieve_time_filter.append(sample['id'])
                filtered = True
            if sample['submission_has_media']:
                media_filter.append(sample['id'])
                filtered = True
            if not filtered:
                not_filtered.append(sample['id'])

        all_filters = list(set(list(edit_filer) + list(link_filter) + list(retrieve_time_filter) + list(media_filter)))
        print(f"edit_filer: {len(edit_filer)} - {len(edit_filer)/len(data_to_save)*100}%\nlink_filter: {len(link_filter)} - {len(link_filter)/len(data_to_save)*100}%\nretrieve_time_filter: {len(retrieve_time_filter)} - {len(retrieve_time_filter)/len(data_to_save)*100}%\nmedia_filter: {len(media_filter)} - {len(media_filter)/len(data_to_save)*100}%\nall_filters: {len(all_filters)} - {len(all_filters)/len(data_to_save)*100}%")
        with open(path_out+subreddit+"_filters.json", "w") as file:
            json.dump({"edit_filer": edit_filer, "link_filter": link_filter, "retrieve_time_filter": retrieve_time_filter, "media_filter": media_filter, "all_filters": all_filters}, file)

        data_to_save = [data_to_save[d] for d in not_filtered]
        with open(path_to_save, "wb") as file:
            pickle.dump(data_to_save, file)
        print(f"saved {len(data_to_save)} processed to {path_to_save} ({ignored_due_to_time_diff} ignored)")
