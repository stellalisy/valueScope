import matplotlib.pyplot as plt
import os
import statistics
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from collections import Counter

subreddit = "askwomen"
subreddit = "askmen"

dimension = "formality"
# dimension = "humor"
# dimension = "length" 
dimension = "politeness" 
dimension = "sarcasm"
dimension = "supportiveness" 


plot_delta = True
plot_zscore = True

data_dir = "/gscratch/argon/stelli/reddit_norm/upvote_prediction/data/output"
output_dir = "/gscratch/argon/stelli/reddit_norm/rpm_model/rpm_figures"

if plot_delta:
    import json
    with open(os.path.join(data_dir, f"upvote_info_{subreddit}_{dimension}.jsonl"), "r") as json_file:
        json_list = list(json_file)
    data = [json.loads(json_str) for json_str in json_list]
    norm_ratings, upvotes = [], []
    for sample in data:
        orig_rating = sample["original_rating"]
        if not plot_zscore:
            orig_upvote = sample["original_score"] if "original_score" in sample else sample["original_comment"]["vote"]
        else:
            orig_upvote = sample["original_comment"]["zscore"]
        for r in range(1, 6):
            upvotes.append(sample[str(r)]["zscore"] - orig_upvote if plot_zscore else sample[str(r)]["vote"] - orig_upvote)
            norm_ratings.append(sample[str(r)]["rating"] - orig_rating)

else:
    with open(os.path.join(data_dir, f"pred_rating_{subreddit}_{dimension}.txt"), "r") as f:
        norm_ratings = [float(d.strip()) for d in f.readlines()]

    with open(os.path.join(data_dir, f"pred_upvote_{subreddit}_{dimension}.txt"), "r") as f:
        upvotes = [float(d.strip()) for d in f.readlines()]


def func_4th(x, a, b, c, d, e):
    # return a*np.exp(-b*x) + c
    return a + b*x + c*x**2 + d*x**3 + e*x**4

def func_5th(x, a, b, c, d, e, f):
    # return a*np.exp(-b*x) + c
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5

def func_lin(x, a, b):
    # return a*np.exp(-b*x) + c
    return a + b*x

def make_rpm(x_axis, y_axis, dimension, subreddit, fig_name=None, linear=True):
    # y_std = statistics.stdev(y_axis[dimension])
    # y_plot = np.log2([y if y > 0 else 1 for y in y_axis[dimension]])
    y_plot = np.array(y_axis)
    x_plot = np.array(x_axis)

    # y_temp = pd.DataFrame(y_axis[dimension])
    # print("outlier indices:", (np.abs(stats.zscore(y_plot)) < 3))

    outlier_indices = (np.abs(stats.zscore(y_plot)) < 1)
    # outlier_indices = y_plot < 40
    y_plot = y_plot[outlier_indices]
    x_plot = x_plot[outlier_indices]

    x_counter = Counter(x_plot)
    for key, value in x_counter.items():
        if value < 10:
            indices = np.where(x_plot == key)
            x_plot = np.delete(x_plot, indices)
            y_plot = np.delete(y_plot, indices)

    # outlier_indices = np.where((y_plot['bmi'] > 0.12) & (y_plot['bp'] < 0.8))
 
    # no_outliers = y_plot.drop(outlier_indices[0])
    
    if linear:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_plot,y_plot)
        predict_y = intercept + slope * x_plot
        plt.plot(x_plot, predict_y, 'r-', label=f'r = {r_value:.2f}, m = {slope:.2f}')
        # plt.legend(prop={'size': 18})

    else:
        func = func_5th
        popt, pcov = curve_fit(func, x_plot, y_plot)
        print("popt: ", popt)
        print("pcov: ", pcov)
        x_fit = np.linspace(-1, 1, 50)
        plt.plot(x_fit, func(x_fit, *popt), 'r-', label='fitted curve')

    plt.scatter(x_plot, y_plot, s=0.5)
    plt.title(f'{dimension} in {subreddit}' + " (delta)" if plot_delta else f'')
    plt.ylabel("upvote zscores" if plot_zscore else "upvotes")
    plt.xlabel("norm intensity")
    if fig_name is not None:
        plot_title = fig_name
    else:
        if plot_delta and plot_zscore:
            plot_title = f'{dimension}_{subreddit}_delta_zscore.png'
        elif plot_delta and not plot_zscore:
            plot_title = f'{dimension}_{subreddit}_delta.png'
        elif not plot_delta and plot_zscore:
            plot_title = f'{dimension}_{subreddit}_zscore.png'
        else:
            plot_title = f'{dimension}_{subreddit}.png'
    
    plt.legend()
    plt.savefig(os.path.join(output_dir, plot_title), dpi=200)
    plt.show()
    plt.clf()


make_rpm(norm_ratings, upvotes, dimension, subreddit)