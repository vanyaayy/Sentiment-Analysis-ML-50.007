# Sentiment Analysis with Random Forest (using scikit-learn):


import numpy as np

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
def create_bow_representation(texts):
    vocab = {}
    bow_matrix = np.zeros((len(texts), len(vocab)))

    for idx, text in enumerate(texts):
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
            word_idx = vocab[word]
            bow_matrix[idx, word_idx] += 1

    return bow_matrix

train_features = create_bow_representation(train_texts)
test_features = create_bow_representation(test_texts)

# Train a Random Forest classifier
class RandomForestClassifier:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            model = self._create_tree()
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

    def _create_tree(self):
        # Create a simple decision tree
        pass

model = RandomForestClassifier(n_estimators=100)
model.fit(train_features, train_labels)

# Predict and evaluate
predictions = model.predict(test_features)
rounded_predictions = np.round(predictions)  # Assuming binary classification
accuracy = np.mean(rounded_predictions == test_labels)
print(f"Accuracy: {accuracy:.2f}")
