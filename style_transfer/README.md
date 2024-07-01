# Synthetic Data Generation for Normness Distillation and Community Preference Distillation

This repository contains the code to generate synthetic data for studying normative behaviors and community preferences on Reddit. The synthetic data is generated using the `transformers` library to leverage language models for creating controlled variations in Reddit comments along different norm dimensions.

## Table of Contents
- [Usage](#usage)
- [Explanation](#explanation)
- [Example](#example)


## Usage
To generate synthetic data, run the script with the appropriate arguments. The arguments are specified below:

```sh
python generate_synthetic_data.py -r <subreddit> -n <norm_dimension> [-d <data_dir>] [-o <output_dir>] [-m <model_name>] [-u <upvote_dir>] [-b <batch_size>] [-sb <sub_batch_size>] [-q <quantization>] [-s <start_idx>] [-e <end_idx>]
```

### Arguments
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

### Example
```sh
python generate_synthetic_data.py -r askmen -n formality
```

## Explanation
The script generates synthetic data for studying normative behaviors and community preferences on Reddit by varying a single norm dimension in Reddit comments. The process involves the following steps:

1. **Argument Parsing**: The script accepts various arguments to configure the generation process.
2. **Model Loading**: Loads the specified language model with optional quantization.
3. **Data Loading**: Loads the Reddit comments and prepares the lookup tables for comment IDs and submission titles.
4. **Comment Transformation**: Transforms each comment by varying one norm dimension and generates multiple versions of the comment.
5. **Response Generation**: Uses the language model to generate the transformed comments and obtain normness ratings.
6. **Output Writing**: Writes the transformed comments to the specified output directory.


## Normness Distillation
The normness distillation stage aims to address two key challenges:
1. Recreate "hypothetical conditions" where individuals consider alternative behaviors.
2. Mitigate confounding factors such as content variations and personal linguistic habits by varying only one norm dimension at a time.

## Community Preference Distillation
The community preference distillation stage isolates the effects of specific norm dimensions on community reactions by calculating the difference in predicted preferences between the original and rewritten comments.

For more details on the purpose and methodology, refer to the relevant sections in the code and paper.

## License
This project is licensed under the MIT License.
