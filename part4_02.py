import numpy as np

# Helper function to preprocess the text
def preprocess_text(text):
    # Implement your text preprocessing here (e.g., lowercase, removing special characters)
    return text

# Helper function to calculate Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2)**2))

class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return predictions

# Load and preprocess the training data for sentiment analysis
train_data = []
with open("Data/RU/train", "r", encoding="utf-8") as train_file:
    words = []
    labels = []
    for line in train_file:
        line = line.strip()
        if line:
            word, label = line.rsplit(' ', 1)
            words.append(preprocess_text(word))
            labels.append(label)
        else:
            if words:
                train_data.append((words, labels))
                words = []
                labels = []

# Create a vocabulary from the training data
vocab = set()
for words, _ in train_data:
    vocab.update(words)
vocab = list(vocab)
label_to_int = {label: i for i, label in enumerate(set(label for _, labels in train_data for label in labels))}
int_to_label = {i: label for label, i in label_to_int.items()}

# Convert sentiment labels to integers
y_train = [label_to_int[label] for _, labels in train_data for label in labels]

# Convert text data into feature vectors using bag-of-words representation
def text_to_bow(words):
    bow_vector = np.zeros(len(vocab))
    for word in words:
        if word in vocab:
            bow_vector[vocab.index(word)] += 1
    return bow_vector
X_train = np.array([text_to_bow(words) for words, _ in train_data])

# Load and preprocess the development data
dev_tweets = []
with open("Data/RU/dev.p4.out", "r", encoding="utf-8") as dev_file:
    for line in dev_file:
        processed_tweet = preprocess_text(line.strip())
        dev_tweets.append(processed_tweet)

# Convert development data to bag-of-words feature vectors
X_dev = np.array([text_to_bow(tweet) for tweet in dev_tweets])

# Instantiate and train the KNN classifier
k = 5  # Number of neighbors
knn_classifier = KNNClassifier(k)
knn_classifier.fit(X_train, y_train)

# Predict sentiments on the development set using KNN classifier
predicted_sentiments_knn = knn_classifier.predict(X_dev)

# Write predicted sentiments to the output file
output_file_path = "Data/RU/dev_knn.out"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    for i, sentiment in enumerate(predicted_sentiments_knn):
        predicted_label = int_to_label[sentiment]
        if dev_tweets[i] != "":
            output_file.write(f"{dev_tweets[i]} {predicted_label}\n")
        else:
            output_file.write("\n")

print(f"Predictions written to {output_file_path}")
