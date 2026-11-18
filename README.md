# Comparing Rule-Based, Machine Learning, and LLM Approaches for Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.13.9-blue)
![NLP](https://img.shields.io/badge/NLP-Sentiment_Analysis-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![LLM](https://img.shields.io/badge/LLM-Zero--Shot-purple)

## Overview

This project compares three sentiment analysis approaches on the same dataset:

- **VADER** (Rule-Based NLP)
- **NLTK + Naïve Bayes** (Classical Machine Learning)
- **LLM Zero-shot Classification** (Claude/OpenAI)

The objective is to evaluate each method not only on prediction quality but also on practical considerations such as inference speed and cost.

---

## Research Question

> Which sentiment analysis approach offers the best trade-off between accuracy, robustness, speed, and cost?

---

## Project Structure

```text
sentiment-project/
│
├── README.md
├── requirements.txt
├── prompts.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_vader.ipynb
│   ├── 03_classic_ml.ipynb
│   ├── 04_llm.ipynb
│   └── 05_compare.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── vader_model.py
│   ├── ml_model.py
│   ├── llm_model.py
│   ├── evaluation.py
│   └── utils.py
│
├── results/
│   ├── metrics.csv
│   ├── leaderboard.csv
│   ├── llm_cache.jsonl
│   └── charts/
│
└── tests/
```

---

## Dataset

- IMDB Reviews

Sentiment Labels:

- Positive
- Neutral
- Negative

---

## Methodology

### 1. Data Preparation

Notebook:

```text
01_data.ipynb
```

Tasks:

- Load dataset
- Perform Exploratory Data Analysis (EDA)
- Visualize class distribution
- Remove duplicates
- Clean text
- Train/Test Split

---

### 2. Rule-Based Sentiment Analysis

Notebook:

```text
02_vader.ipynb
```

Model:

```text
VADER
```

Advantages:

- No training required
- Extremely fast
- Free

Limitations:

- Weak with sarcasm
- Weak with contextual sentiment

---

### 3. Classical Machine Learning

Notebook:

```text
03_classic_ml.ipynb
```

Models evaluated:

- Naïve Bayes
- Logistic Regression
- Linear SVM

Features:

```text
TF-IDF
```

Expected strongest baseline:

```text
Logistic Regression / Linear SVM
```

---

### 4. LLM Sentiment Classification

Notebook:

```text
04_llm.ipynb
```
- Claude Haiku

