filter_phrases = ['I apologize, but','not able to fulfill this request','cannot fulfill your request', "I cannot provide a rewritten", "rewritten comment"]
split_phrases = ['"\n\nThis', '"\n\nRating:', '"\n\nPlease', '"\n\nNote', '"\n\nRewritten', '"\n\nOriginal', '"\n\nRemember', '"\n\nI rewrote', '"\n\nI hope', '"\n\nI rated', '"\n\nI Changed', '"\n\nI kept', '"\n\nI tried', '"\n\nThe original', '"\n\nI\'ve']
def filter_comment(comment):
    if any([phrase in comment for phrase in filter_phrases]):
        return None
    for split_phrase in split_phrases:
        if split_phrase in comment:
            return comment.split(split_phrase)[0]
    return comment
    

def process_data(df):
    _data = []
    for _, row in df.iterrows():
        original_id = row['id']
        _data.append({'id': original_id+"-0", 'id-original': original_id,'comment': row['original_comment'], 'rating': row['original_rating'], 'post_title': row['submission_title']})
        for scale in ['1','2','3','4','5']:
            filtered_comment = filter_comment(row[scale])
            if filtered_comment is not None:
                _data.append({'id': original_id+"-"+scale,'id-original':original_id, 'comment': filtered_comment, 'rating': int(scale), 'post_title': row['submission_title']})
    return _data