# Comparing Rule-Based, Machine Learning, and LLM Approaches for Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.13.9-blue)
![NLP](https://img.shields.io/badge/NLP-Sentiment_Analysis-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![LLM](https://img.shields.io/badge/LLM-Zero--Shot-purple)

## Overview

This project compares three sentiment analysis approaches on the same dataset:

- **VADER** (Rule-Based NLP)
- **NLTK + Naïve Bayes** (Classical Machine Learning)
- **LLM Zero-shot Classification** (Claude)

The objective is to evaluate each method not only on prediction quality but also on practical considerations such as inference speed and cost.

---

## Research Question

> Which sentiment analysis approach offers the best trade-off between accuracy, robustness, speed, and cost?

---

## Dataset

- IMDB_Dataset

Sentiment Labels:

- Positive
- Neutral
- Negative

---

## Methodology

### 1. Data Preparation


Tasks:

- Load dataset
- Perform Exploratory Data Analysis (EDA)
- Visualize class distribution
- Remove duplicates
- Clean text
- Train/Test Split

---

### 2. Rule-Based Sentiment Analysis

Models evaluated:

- VADER

Advantages:

- No training required
- Extremely fast
- Free

Limitations:

- Weak with sarcasm
- Weak with contextual sentiment

---

### 3. Classical Machine Learning

Models evaluated:

- Naïve Bayes

Approach
- Bag of words
---

### 4. LLM Sentiment

Model Evaluated
- Claude Haiku

Approach
- Prompt-based sentiment classification

