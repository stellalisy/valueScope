from collections import Counter

# with open("/gscratch/cse/stelli/reddit_norm/data/norms/llm_prompt_norms_round_2.txt", "r") as f:
#     lines = f.readlines()

# all_norms = []
# for line in lines:
#     if ":" not in line:
#         continue
#     all_norms.append(line.strip().replace("...", "").split(":")[0].split(". ")[-1])

# print("\n".join(Counter(all_norms).keys()))
# print(len(Counter(all_norms)))

# with open("/gscratch/cse/stelli/reddit_norm/data/norms/classes_1.txt", "r") as f:
#     lines = f.readlines()

# all_classes = []
# for line in lines:
#     all_classes.append(line.strip())

# print("\n".join(Counter(all_classes).keys()))
# print(len(Counter(all_classes)))

with open("/gscratch/cse/stelli/reddit_norm/data/norms/target_classes.txt", "r") as f:
    target_classes = [s.strip() for s in f.readlines()]


with open("/gscratch/cse/stelli/reddit_norm/data/norms/classes_1.txt", "r") as f:
    lines = f.readlines()

classes = []
for line in lines:
    class_ = line.strip().split("\t")[1]
    if class_ in target_classes:
        classes.append(class_)


# print("\n".join(Counter(classes).keys()))
print(Counter(classes))
class_counter = Counter(classes)

class_counter_sorted = sorted(class_counter.items(), reverse=True, key=lambda x:x[1])
for k, v in class_counter_sorted:
    print("{}".format(v))

print(len(Counter(classes)))