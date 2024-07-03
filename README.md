# ValueScope: Unveiling Implicit Norms and Values via Return Potential Model of Social Interactions

## Project Overview
VALUEscope is a computational framework that leverages language models to quantify social norms and values within online communities. The project is grounded in social science perspectives on normative structures and utilizes a novel Return Potential Model (RPM) to dissect and analyze linguistic and stylistic expressions across various digital platforms.

## Repository Structure
```
valueScope/
│
├── norm_induction/               # Scripts and data for norm induction processes
│   └── data/                     # Data and outputs from the norm induction process
│   └── scripts/                  # Scripts related to surface important norms
├── norm_prediction/              # Scripts for training norm prediction models and using them for binary label generation
│   └── human-annotation-labels/  # JSON files of 4 topics for 450 human-annotated labels
│   └── synthetic-labels/         # Scripts and prompts for GPT synthetic label generations for classifier training
├── style_transfer/
│   └── scripts/                  # Scripts related to style transfer generation
│   └── filter/                   # Scripts related to style transfer filter
├── upvote_prediction/
│   └── scripts/                  # Scripts for predicting upvotes based on community norms
├── .gitignore                    # Specifies intentionally untracked files to ignore
└── README.md                     # This README file
```

## Installation Instructions
To set up the VALUEscope framework on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/stellali7/valueScope.git
   ```
2. **Navigate to the repository directory:**
   ```bash
   cd valueScope
   ```

## Usage
The framework includes several components that can be run separately. Instructions for running each component are provided in their respective directories.

## License
This project is licensed under the MIT License.




## Citation
If you use VALUEscope in your research, please consider citing our work:

```bibtex
@misc{park2024valuescope,
      title={ValueScope: Unveiling Implicit Norms and Values via Return Potential Model of Social Interactions}, 
      author={Chan Young Park and Shuyue Stella Li and Hayoung Jung and Svitlana Volkova and Tanushree Mitra and David Jurgens and Yulia Tsvetkov},
      year={2024},
      eprint={2407.02472},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.02472}, 
}
```