# Synthetic Data Generation and Filtering Pipeline for Normness Distillation and Community Preference Distillation

This repository contains the code to generate synthetic data for studying normative behaviors and community preferences on Reddit (see directory `scripts`). The synthetic data is generated using the `transformers` library to leverage language models for creating controlled variations in Reddit comments along different norm dimensions. This repository also contains relevant code for our synthetic data filtering pipeline, including preprocessing, lexical, fluency, and content preservation filters, to ensure data quality (see directory `filter`).


## Table of Contents
- [Synthetic Data Generation](#synthetic-data-generation)
    - [Usage](#usage)
    - [Example](#example)
    - [Explanation](#explanation)
- [Synthetic Data Filtering Pipeline](#synthetic-data-filtering-pipeline)

## Synthetic Data Generation

### Usage
To generate synthetic data, run the script with the appropriate arguments. The arguments are specified below:

```sh
python generate_synthetic_data.py -r <subreddit> -n <norm_dimension> [-d <data_dir>] [-o <output_dir>] [-m <model_name>] [-u <upvote_dir>] [-b <batch_size>] [-sb <sub_batch_size>] [-q <quantization>] [-s <start_idx>] [-e <end_idx>]
```

#### Arguments
- `-r`, `--subreddit`: The target subreddit (required).
- `-n`, `--norm_dimension`: The norm dimension to vary (required).
- `-d`, `--data_dir`: Directory of the processed upvote data (default: `/gscratch/argon/stelli/reddit_norm/upvote_prediction/data/processed_upvotes/`).
- `-o`, `--output_dir`: Directory to save the output data (default: `/gscratch/argon/stelli/reddit_norm/style_transfer/data/output/llama3/`).
- `-m`, `--model_name`: Model name for the language model (default: `meta-llama/Meta-Llama-3-8B-Instruct`).
- `-u`, `--upvote_dir`: Directory for upvote prediction models (default: `None`).
- `-b`, `--batch_size`: Batch size for processing comments (default: 1).
- `-sb`, `--sub_batch_size`: Sub-batch size for processing comments (default: 6).
- `-q`, `--quantization`: Quantization bits (default: `None`).
- `-s`, `--start_idx`: Start index for processing comments (default: 0).
- `-e`, `--end_idx`: End index for processing comments (default: 100000000).

#### Example
```sh
python generate_synthetic_data.py -r askmen -n formality
```

### Explanation
The script generates synthetic data for studying normative behaviors and community preferences on Reddit by varying a single norm dimension in Reddit comments. The process involves the following steps:

1. **Argument Parsing**: The script accepts various arguments to configure the generation process.
2. **Model Loading**: Loads the specified language model with optional quantization.
3. **Data Loading**: Loads the Reddit comments and prepares the lookup tables for comment IDs and submission titles.
4. **Comment Transformation**: Transforms each comment by varying one norm dimension and generates multiple versions of the comment.
5. **Response Generation**: Uses the language model to generate the transformed comments and obtain normness ratings.
6. **Output Writing**: Writes the transformed comments to the specified output directory.


## Synthetic Data Filtering Pipeline

For this pipeline, we provided a jupyter notebook `st_filter.ipynb`, which contains documentation and code for each of the filters in our pipeline. The workflow across each filter is denoted in markdown headers and can be summarized as follows:

1. Loading the generated synthetic data
2. Preprocessing Filter 
3. Lexical Filter
4. Fluency Filter
5. Content Preservation Filter

For the Fluency Filter and Content Preservation Filter, we need to compute the perplexity and BERTScore metrics, respectively, to filter the synthetic data. Thus, we use SLURM to send GPU jobs and compute these metrics. The code to queue SLURM jobs are within the notebook and the scripts to compute these metrics are contained in `style_transfer/filter/scripts/` (e.g. `bert-score-compute.py` and `perplexity_compute.py`). Note that `perplerxity_compute.py` requires an access token for using `microsoft/DialoGPT-medium` from HuggingFace. 

The notebook cells can be run sequentially from beginning to end. However, for the Fluency and Content Preservation Filters, you must wait for the SLURM jobs to complete and the metrics to be computed before executing the subsequent notebook cells, which uses the computed metrics to filter out synthetic data.

The current implementation saves all the perplexity and BERTScore computations in a directory and resumes where it left off (e.g. due to crash, new synthetic data generation), thus avoiding any redundant computes. The current implementation will only queue jobs if there are newly generated synthetic data in the pipeline that have not been filtered yet. You may need to adjust the notebook based on your experiments (e.g. directories, norm dimensions, reddits, etc). 


## Normness Distillation
The normness distillation stage aims to address two key challenges:
1. Recreate "hypothetical conditions" where individuals consider alternative behaviors.
2. Mitigate confounding factors such as content variations and personal linguistic habits by varying only one norm dimension at a time.

## Community Preference Distillation
The community preference distillation stage isolates the effects of specific norm dimensions on community reactions by calculating the difference in predicted preferences between the original and rewritten comments.

For more details on the purpose and methodology, refer to the relevant sections in the code and paper.

## License
This project is licensed under the MIT License.
