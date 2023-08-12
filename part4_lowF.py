import numpy as np

# Helper function to preprocess the text
def preprocess_text(text):
    # Implement your text preprocessing here (e.g., lowercase, removing special characters)
    return text

# Load and preprocess the training data for sentiment analysis
train_data = []
with open("Data/RU/train", "r", encoding="utf-8") as train_file:
    for line in train_file:
        line = line.strip()
        if line:
            word, label = line.rsplit(' ', 1)
            processed_word = preprocess_text(word)
            train_data.append((processed_word, label))

# Create a vocabulary from the training data
vocab = set()
for word, _ in train_data:
    words = word.split()
    vocab.update(words)
vocab = list(vocab)
label_to_int = {label: i for i, label in enumerate(set(label for _, label in train_data))}
int_to_label = {i: label for label, i in label_to_int.items()}

# Convert sentiment labels to integers
y_train = [label_to_int[label] for _, label in train_data]

# Convert text data into feature vectors using bag-of-words representation
def text_to_bow(text):
    bow_vector = np.zeros(len(vocab))
    words = text.split()
    for word in words:
        if word in vocab:
            bow_vector[vocab.index(word)] += 1
    return bow_vector
X_train = np.array([text_to_bow(tweet) for tweet, _ in train_data])

class NaiveBayesClassifier:
     def __init__(self, num_classes):
        self.num_classes = num_classes
        self.class_counts = np.zeros(num_classes)
        self.word_counts = np.zeros((num_classes, len(vocab)))

     def fit(self, X, y):
        for i in range(len(X)):
            class_label = int(y[i])
            self.class_counts[class_label] += 1
            self.word_counts[class_label] += X[i]

     def predict(self, X):
        predictions = []
        for i in range(len(X)):
            likelihoods = np.log((self.word_counts + 1) / (self.class_counts[:, np.newaxis] + len(vocab)))
            scores = np.sum(likelihoods * X[i], axis=1)
            predicted_class = np.argmax(scores)
            predictions.append(predicted_class)
        return predictions
    # ... (rest of the class implementation)

# Instantiate and train the Naive Bayes classifier
num_classes = len(set(label for _, label in train_data))
nb_classifier = NaiveBayesClassifier(num_classes)
nb_classifier.fit(X_train, y_train)

# Load and preprocess the development data
dev_tweets = []
with open("Data/RU/dev.in", "r", encoding="utf-8") as dev_file:
    for line in dev_file:
        processed_tweet = preprocess_text(line.strip())
        dev_tweets.append(processed_tweet)

# Convert development data to bag-of-words feature vectors
X_dev = np.array([text_to_bow(tweet) for tweet in dev_tweets])

# Predict sentiments on the development set
predicted_sentiments = nb_classifier.predict(X_dev)

# Write predicted sentiments to the output file
with open("Data/RU/dev.p4.out", "w", encoding="utf-8") as output_file:
    for i, sentiment in enumerate(predicted_sentiments):
        predicted_label = int_to_label[sentiment]
        if dev_tweets[i] != "":
            output_file.write(f"{dev_tweets[i]} {predicted_label}\n")
        else:
            output_file.write("\n")
