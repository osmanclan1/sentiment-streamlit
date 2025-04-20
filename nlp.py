# lets make an nlp classifier that simply detects if a text is positive or negative
# using logistic regression is just like linear regression but with a sigmoid function

# lets first import the libraries we need
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import joblib


# === Load data ===
# load the dataset
dataset = load_dataset("imdb")
# this dataset contains 50k reviews from IMDB
train_texts = dataset['train']['text'] # this set contains the training data
train_labels = dataset['train']['label'] # this set contains the labels

# we will use the test set for validation
test_texts = dataset['test']['text'] # this set contains the test data
test_labels = dataset['test']['label'] # this set contains the labels

# ok now we have the data
# === Basic text cleaning (optional here) ===
# convert the text to string
train_texts = [str(text) for text in train_texts]
test_texts = [str(text) for text in test_texts]
# convert the labels to int
train_labels = [int(label) for label in train_labels]
test_labels = [int(label) for label in test_labels]
# === TF-IDF Vectorization ===
# create the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
# fit the vectorizer on the training data
X = vectorizer.fit_transform(train_texts)
# transform the test data
X_test = vectorizer.transform(test_texts)

# create the labels
y = np.array(train_labels)
y_test = np.array(test_labels)

# === Train-validation split ===
# split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# this will give us 80% of the data for training and 20% for validation



# === Logistic Regression (Baseline) ===
# create the model

model = LogisticRegression(max_iter=1000)

# train the model
model.fit(X_train, y_train)

# make predictions
val_preds = model.predict(X_val)

# evaluate the model
print("Validation Accuracy:", accuracy_score(y_val, val_preds))

# print accuracy score
# === Predict on test set ===
test_preds = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))



# Save the model and vectorizer
joblib.dump(model, "logistic_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

