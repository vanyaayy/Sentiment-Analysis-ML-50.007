# Sentiment Analysis with Random Forest (using scikit-learn):


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
def preprocess_data(data_path):
    texts, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().rsplit(' ', 1)
            texts.append(text)
            labels.append(int(label))
    return texts, labels

train_texts, train_labels = preprocess_data("train.txt")
test_texts, test_labels = preprocess_data("test.txt")

# Create Bag-of-Words representations
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_texts)
test_features = vectorizer.transform(test_texts)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_features, train_labels)

# Predict and evaluate
predictions = model.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")
