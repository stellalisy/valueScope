# Norm Prediction Readme

## Directory Overview

This directory contains the files and scripts needed to train norm prediction models and to use them for binary label generation, and ultimately combining them to compute the final winrate. 

## Directory Structure
```
norm_prediction/
│
├── human-annotation-labels/
│ ├── askscience_binary.json
│ ├── finance_binary.json
│ ├── gender_binary.json
│ └── politics_binary.json
│
├── synthetic-labels/
│ ├── gpt-synthetic-label-generation.ipynb
│ └── prompts.py
│
├── infer.py
├── label2winrate.py
└── train.py
```

## File Descriptions

### human-annotation-labels
This directory contains JSON files of four topics for 450 human-annotated labels. Each file includes annotations for five norm dimensions, such as formality, supportiveness, sarcasm, politeness, and humor.

Example structure:
```json
{
    "title1": "How is a tree trunk able to grow \"through\" an intervening obstacle such as a fence?",
    "description1": "...",
    "comment1": "...",
    "title2": "Can extensive use of deep geothermal heat...",
    "description2": "...",
    "comment2": "...",
    "annotation": {
        "formality": "2",
        "supportiveness": "1",
        "sarcasm": "2",
        "politeness": "1",
        "humor": "1"
    }
}
```
### synthetic-labels
This directory contains two files (e.g. gpt-synthetic-label-generation.ipynb, prompts.py) relevant to generating synthetic labels via GPT-4, which can then be used to train a smaller classifier model. The notebook `gpt-synthetic-label-generation.ipynb` holds in-depth documentation and directions explaining the workflow. The workflow within the notebook are denoted with markdown headers and can be summarized as:

1. Sampling random comments from Reddit Data Dumps
2. Employing GPT3.5 to rate comments for stratified sampling
3. Creating random pairwise comparisons among the sampled comments
4. Generate synthetic labels for these pairs using GPT-4

Note that `prompts.py` contains all the prompt-related texts that are used to generate the synthetic labels within gpt-synthetic-label-generation.ipynb.

### train.py
This script is used to train the models on the provided data. It processes the input data, initializes the model, and runs the training loop.

Usage:
```bash
python train.py <NORM_DIMENSION> <CATEGORY> [SEED]
```

### infer.py
This script is used to infer the labels using the trained models. It loads the model and performs inference on the given data.

Usage:
```bash
python infer.py <TARGET_DIMENSION> <TARGET_TOPIC> [BATCH_SIZE] [BOOL_FILTER] [BOOL_PRIORITIZE_LESS]
```

### label2winrate.py

This script is used to convert binary labels to win rates. It processes the results from the inference and computes the win rates.

Usage:
```
python label2winrate.py <TARGET_CAT>
```

## Example Workflow

0. **Generate synthetic labels**: Generate synthetic labels via GPT-4 to train a smaller model
Adjust code (e.g. directories, your variables, etc) and execute cells in `norm_prediction/synthetic-labels/gpt-synthetic-label-generation.ipynb`

1. **Training**: Train a model on a specific norm dimension and category.
```bash
python train.py formality science
```

2. **Inference**: Use the trained model to predict labels on new data.
```bash
python infer.py formality science 64 True True
```

3. **Win Rate Calculation**: Convert the predicted binary labels to win rates.
```bash
python label2winrate.py science
```