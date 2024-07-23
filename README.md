# Fake News Detection

This project is aimed at detecting fake news using a machine learning model trained with a Support Vector Machine (SVM) algorithm. The goal is to classify news articles as either "Real News" or "Fake News" based on their content.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model](#model)

## Introduction

Fake news has become a significant issue in today's digital age. The purpose of this project is to build a reliable fake news detection system using Natural Language Processing (NLP) and machine learning techniques. The system utilizes an SVM model for classification.

## Dataset

The dataset used for this project consists of news articles labeled as real or fake. The dataset contains various features such as the title, author, and main text of the articles.

## Preprocessing

The preprocessing steps include:

1. Converting text to lowercase.
2. Removing punctuation and special characters.
3. Expanding contractions (e.g., "I'm" to "I am").
4. Lemmatization to reduce words to their base form.
5. Removing stopwords to eliminate common but uninformative words.

We use the NLTK library for text preprocessing tasks.

## Model

The model used in this project is a Support Vector Machine (SVM) classifier. SVM is a supervised learning algorithm that can be used for both classification and regression challenges. However, it is mostly used in classification problems. In this project, we train the SVM model using a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer to convert the text data into numerical form.

