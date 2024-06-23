with open("/gscratch/cse/stelli/reddit_norm/data/subreddits/temp.txt", "r") as f:
    lines = f.readlines()


subreddits = set()
for line in lines:
    if ":" not in line:
        continue
    subreddits.update(line.strip().replace("...", "").split(":")[-1].split(", "))
print("total number of subreddits covered:", len(subreddits))