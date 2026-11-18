# Uplift B3 Milestone 4 - Sentiment Analysis

## Overview

This project compares classical NLP sentiment models with a prompt-based Anthropic Claude approach using the IMDb movie review dataset.

The analysis is split across two notebooks:

- `01_classic_vader_naive_bayes.ipynb` evaluates VADER and an NLTK Naive Bayes classifier.
- `02_claude_nlp.ipynb` evaluates Claude Haiku, with VADER included as a comparison.

## Dataset

`IMDB_Dataset.csv` contains 50,000 movie reviews and two columns:

- `review`: the text of the IMDb review
- `sentiment`: the human sentiment label, `positive` or `negative`

The dataset is balanced, with 25,000 reviews in each sentiment class. There are no missing values.

## Notebook 1: Classical and Lexicon-Based Models

### Preprocessing

- Unescapes HTML entities
- Removes HTML tags, URLs, email addresses, mentions, and hashtags
- Retains useful punctuation and sentiment-related words
- Copies reviews to a `clean_review` column

The notebook uses a stratified 80/20 split:

- 40,000 training reviews
- 10,000 test reviews

### Models

#### VADER

VADER assigns sentiment from a compound polarity score using a positive threshold of `0.2` and a negative threshold of `-0.2`. The resulting labels are `negative`, `neutral`, or `positive`.

#### NLTK Naive Bayes

The classifier uses:

- `TweetTokenizer`
- Negation marking with `mark_negation`
- English stop-word removal while preserving words such as `not`, `nor`, `no`, and `very`
- Word-presence features from the 3,000 most frequent training tokens

### Classical Model Results

| Model | Accuracy | Macro-F1 (negative/positive) |
|---|---:|---:|
| VADER | 68.30% | 68.40% |
| NLTK Naive Bayes | 83.34% | 83.28% |

The NLTK Naive Bayes model outperformed VADER on both accuracy and binary macro-F1. For the human-labelled negative and positive classes, it achieved 83.34% accuracy and 83.28% macro-F1. The VADER macro-F1 value comes from the notebook's `evaluate()` function, which calculates it for only `negative` and `positive`; VADER's printed report also shows a separate macro average that includes its neutral prediction class.

## Notebook 2: Claude Haiku Sentiment Classification

The Claude notebook loads the full 50,000-review dataset, then randomly samples 300 reviews with `random_state=1` for prompt-based classification.

Claude is instructed to return exactly one lowercase label:

- `positive`
- `neutral`
- `negative`

The function sends each review to the model configured as `claude-haiku-5-5` and normalises unexpected responses to `neutral`.

### Claude Results

On the 300 sampled reviews:

- Accuracy: 92.00%
- Macro F1 for `negative` and `positive`: 95.00%

The notebook also compares Claude with VADER through cross-tabulation and produces `Sentiment_Analysis_IMDB_Dataset_with_claude.csv`, containing the original review and sentiment columns plus:

- `claude_sentiment`
- `vader_sentiment`

## Results Table

| Model | Accuracy | Macro-F1 (negative/positive) | Hard-cases accuracy | Sec / 1k preds | Cost / 1k preds |
|---|---:|---:|---:|---:|---:|
| VADER | 68.30% | 68.40% | Not measured | Not measured | $0 |
| NLTK + Naive Bayes | 83.34% | 83.28% | Not measured | Not measured | ~$0 |
| LLM zero-shot (Claude Haiku) | 92.00%* | 95.00%* | Not measured | Not measured | API usage required |

\* Claude's 92.00% accuracy and 95.00% Macro-F1 for negative and positive classes were calculated on a 300-review sample. The other models were evaluated on the 10,000-review test split.

Neither notebook defines a hard-case subset or measures prediction time, so hard-cases accuracy and seconds per 1,000 predictions are marked as not measured rather than estimated.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- TextBlob
- VADER Sentiment
- WordCloud
- Scikit-learn
- Anthropic Python SDK
- python-dotenv
- Jupyter Notebook

## Setup

```bash
pip install pandas numpy matplotlib seaborn nltk textblob vaderSentiment wordcloud scikit-learn anthropic python-dotenv
```

Download the required NLTK data for stop words:

```python
import nltk
nltk.download("stopwords")
```

For `02_claude_nlp.ipynb`, create a `.env` file in the project directory and set:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

Keep API keys out of notebooks and source control.

## Running the Project

1. Place `IMDB_Dataset.csv` in the project directory.
2. Open `01_classic_vader_naive_bayes.ipynb` and run all cells.
3. Configure `ANTHROPIC_API_KEY`, then run `02_claude_nlp.ipynb`.

## Goal

Compare lexicon-based, classical machine-learning, and LLM-based approaches to determine how well each identifies positive and negative IMDb movie-review sentiment.
