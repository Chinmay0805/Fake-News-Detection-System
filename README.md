# Fake News Detection System

## Overview
The Fake News Detection System is a machine learning web application built using Python and Flask. It is designed to identify whether a news headline is real or fake based on its text content. The system combines text preprocessing, TF-IDF vectorization, and a logistic regression model to make predictions.

---

## 1. Problem Statement

Fake news is a significant issue on digital platforms. The objective of this project is to build a model that can automatically classify news content as real or fake. This is further enhanced with a user-friendly interface built using Flask for seamless interaction.

---

## 2. Tech Stack Used

| Layer              | Technology Used                        |
|-------------------|----------------------------------------|
| Programming        | Python                                 |
| Data Handling      | Pandas, NumPy                          |
| Feature Extraction | Scikit-learn (TF-IDF Vectorizer)       |
| Model              | Logistic Regression                    |
| Evaluation         | Scikit-learn (Confusion Matrix, ROC)   |
| Model Persistence  | Joblib                                 |
| Web Framework      | Flask                                  |
| Visualization      | Matplotlib, Seaborn                    |

---

## 3. Dataset

The dataset contains two important columns:

- Headline → Text input for classification
- Label → 0 = Real News, 1 = Fake News

---

## 4. Data Preprocessing

- Removed missing values
- Lowercased and cleaned all text inputs
- Split the dataset into 80% training and 20% testing sets

---

## 5. Feature Extraction using TF-IDF

Used Term Frequency-Inverse Document Frequency (TF-IDF) to transform textual data into numerical vectors. This method emphasizes important words while reducing the weight of commonly used ones across all documents.

---

## 6. Model Training

The model was trained using Logistic Regression, chosen for its simplicity and strong performance in binary classification tasks. It is particularly effective when used with TF-IDF vectorized data.

---

## 7. Model Evaluation

The model's performance was assessed using:

- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix
- ROC curve

---

## 8. Web Application using Flask

A web application was developed using Flask that allows users to:

- Enter a news headline
- Submit it for prediction
- View the result (Real or Fake)
- Review model evaluation visualizations:
  - Confusion Matrix
  - ROC Curve
  - Classification Report Heatmap

---

## 9. Model Graphs and Reports

When `train_model.py` is executed, the following graphs are automatically generated and saved in the `static/` folder:

- Confusion_Matrix.png
- ROC_Curve.png
- Classification_Report.png

These are then displayed in the result page of the web app.


