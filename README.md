# Spam Email Classification

## Overview

This project focuses on building a machine learning model for spam email classification. The model is trained to differentiate between spam (unwanted or malicious emails) and ham (legitimate emails).

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Custom Input Testing](#custom-input-testing)
- [Dependencies](#dependencies)
- [License](#license)

## Background

Email classification is a common problem in natural language processing and machine learning. In this project, we employ a logistic regression model along with TF-IDF vectorization for feature extraction to classify emails.

## Dataset

The dataset used for training and testing the model consists of a collection of emails labeled as either spam or ham. The dataset has been preprocessed and augmented using Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance.

## Model Training

The logistic regression model is trained on the resampled training data obtained after applying SMOTE. TF-IDF vectorization is utilized to convert email text into numerical features.

## Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and the confusion matrix. This provides insights into the model's ability to correctly classify instances of spam and ham.

## Usage

To use the model for spam email classification, follow these steps:

1. Install the required dependencies (`pip install -r requirements.txt`).
2. Train the model using the provided dataset (`train_model.py`).
3. Evaluate the model's performance (`evaluate_model.py`).
4. Test the model with custom input (`test_custom_input.py`).

## Custom Input Testing

You can test the model with custom input by using the `test_custom_input.py` script. For example:

```python
# Testing the model with custom input
new_email = 'Dear friend, I have a great investment opportunity for you!'
new_email_vectorized = vectorizer.transform([new_email])
prediction = model.predict(new_email_vectorized)
print('Prediction:', prediction)
