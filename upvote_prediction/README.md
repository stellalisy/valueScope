# Upvote Prediction for Reddit Comments

## Overview
This repository includes scripts to preprocess data, train models, and perform inference on models predicting the number of upvotes a Reddit comment might receive. The models are built using the Hugging Face Transformers library, focusing on capturing community engagement through dialogue-based interactions.

## Scripts Overview
- `preprocess_data.py`: Prepares data for training by cleaning and formatting.
- `main.py`: Core script for training and inference.

## Prerequisites
- Python 3.7 or higher
- PyTorch
- Hugging Face Transformers
- NumPy
- Pandas
- scikit-learn
- WANDB for tracking experiments (optional)

## Setup
Install the required Python packages:

```bash
pip install torch transformers numpy pandas scikit-learn wandb
```

## Data Preprocessing
Before training the model, data must be preprocessed. Run the following commands:

```bash
SUBREDDIT=askscience
LABEL_TYPE=label_log
echo "Subreddit: $SUBREDDIT"
echo "Label Type: $LABEL_TYPE"

python "$large/src_upvote_prediction/preprocess_data.py" --subreddit ${SUBREDDIT}
```

## Training the Model
After preprocessing, use the following script to train the model:

```bash
python -u "scripts/src_upvote_prediction/main.py \
    --subreddit $SUBREDDIT \
    --label_type $LABEL_TYPE \
    --sample_eval_size 5000 \
    --model_name "${SUBREDDIT}_log_half_5ep_time_author" \
    --wandb_group $SUBREDDIT \
    --wandb_name "${SUBREDDIT}_log_half_5ep_time_author" \
    --batch_size 8  --train_on_first_half \
    --num_epochs 5  --train \
    --time readable --author all
```

## Inference
For inference using the trained model, execute the following command:

```bash
DIMENSION=humor

python -u "scripts/src_upvote_prediction/main.py" \
    --predict \
    --label_type $LABEL_TYPE \
    --subreddit $SUBREDDIT \
    --dimension $DIMENSION \
    --sample_eval_size 10
```

## Contributing
Contributions are welcome! Please follow the existing code structure and document any changes thoroughly.

## License
Specify the project's license here, or refer to the LICENSE file in the repository.

## Acknowledgments
- Thanks to the developers of the used