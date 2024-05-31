import json

subreddit = "askmen"
file_name_women = f"/gscratch/argon/stelli/reddit_norm/style_transfer/scripts/found_scores_{subreddit}.jsonl"

with open(file_name_women, "r") as json_file:
    json_list = list(json_file)
data = [json.loads(json_str) for json_str in json_list]

x_axis = {"length": [], "formality": [], "supportiveness": [], "sarcasm": [], "politeness": []}
y_axis = {"length": [], "formality": [], "supportiveness": [], "sarcasm": [], "politeness": []}
for comment_info in data:
    for dimension, ratings in comment_info.items():
        if dimension in x_axis:
            identified = [r for r in ratings if r > 0 ]
            if len(identified) == 0: continue
            average = sum(identified)/len(identified)
            x_axis[dimension].append(float(average))
            y_axis[dimension].append(float(comment_info["score"]))

# print("x_axis: ", x_axis)
# print("y_axis: ", y_axis)

import matplotlib.pyplot as plt
import os
import statistics
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

def func(x, a, b, c, d, e):
    # return a*np.exp(-b*x) + c
    return a + b*x + c*x**2 + d*x**3 + e*x**4

for dimension in x_axis:
    # y_std = statistics.stdev(y_axis[dimension])
    # y_plot = np.log2([y if y > 0 else 1 for y in y_axis[dimension]])
    y_plot = np.array(y_axis[dimension])
    x_plot = np.array(x_axis[dimension])

    # y_temp = pd.DataFrame(y_axis[dimension])
    # print("outlier indices:", (np.abs(stats.zscore(y_plot)) < 3))

    outlier_indices = (np.abs(stats.zscore(y_plot)) < 1)
    # outlier_indices = y_plot < 40
    y_plot = y_plot[outlier_indices]
    x_plot = x_plot[outlier_indices]

    # outlier_indices = np.where((y_plot['bmi'] > 0.12) & (y_plot['bp'] < 0.8))
 
    # no_outliers = y_plot.drop(outlier_indices[0])
    
    # for x, y in zip(x_axis[dimension], y_axis[dimension]):

    popt, pcov = curve_fit(func, x_plot, y_plot)
    # print("popt: ", popt)
    # print("pcov: ", pcov)
    x_fit = np.linspace(1, 5, 50)

    plt.plot(x_fit, func(x_fit, *popt), 'r-')
    plt.plot(x_fit, [0]*len(x_fit), 'k-')
    plt.scatter(x_plot, y_plot)
    plt.title(f'{dimension} in {subreddit}')
    plt.ylabel("log # upvotes")
    plt.xlabel("norm intensity")
    plt.savefig(os.path.join("/gscratch/argon/stelli/reddit_norm/style_transfer/data/rpm_figures", f'{dimension}_{subreddit}.png'))
    plt.show()
    plt.clf()