import json

input_filepath = "/gscratch/cse/stelli/reddit_norm/data/subreddits/top5000_sfw.json"
with open(input_filepath, "r") as file:
    data = json.load(file)

names = data.keys()
with open("data/subreddits/top5000_sfw_name_list.txt", "w+") as file:
    file.write("r/" + " r/".join(names))