# Text Representation Matters: A Machine-Learning Study on Women's E-Commerce Clothing Reviews

This project based on Kaggle Women's E-Commerce Clothing Reviews dataset, providing a complete process of feature extraction, data preprocessing, model training and analysis. 

It supports TF-IDF, BERT, and OpenAI embedding and implements feature caching to optimize the subsequent running speed.

## Environment

1. Create a virtual environment and install dependencies
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

2. Environment variable

Please prepare `.env` in the root of project and set the `OPENAI_API_KEY` (For convenient testing, it has been provided).

## Dataset Preparation

1. Go to [Women's E-Commerce Clothing Reviews
](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews/)

2. Download dataset, rename the CSV file to `data.csv`

3. Move `data.csv` to the root of project

## Training Model

When everything ready, your project structure should include:

```bash
tree -a -L 1
.
├── .env
├── .venv
├── README.md
├── analysis.ipynb
├── data.csv
├── extract_features.py
├── requirements.txt
└── train.ipynb
```

Use VSCode or Jupyter Notebook, open `train.ipynb` and select "Run All".

This file will automatically perform data preprocessing, feature extraction, model training and evaluation, and output a summary table of training and testing results.

You can see key information in Jupyter Cells.

Note: the first run might be slow. The features extracted by BERT and OpenAI Embedding will be cached, subsequent runs would be faster.

## Analysis

When training model finished, the project structure should be like this:
```bash
tree -a -L 1
.
├── .env
├── .venv
├── README.md
├── analysis.ipynb
├── bert_embedding_cache.pkl
├── data.csv
├── extract_features.py
├── final.csv
├── openai_embedding_cache.pkl
├── requirements.txt
└── train.ipynb
```

Use VSCode or Jupyter Notebook, open `analysis.ipynb` and select "Run All".

You can see key information in Jupyter Cells.