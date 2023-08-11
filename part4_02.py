import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.features = []  # Store feature vectors of training data
        self.labels = []    # Store corresponding labels

    def train(self, train_path):
        with open(train_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    token, label = line.rsplit(' ', 1)
                    # You need to define how to convert tokens into feature vectors
                    feature_vector = self.convert_token_to_feature_vector(token)
                    self.features.append(feature_vector)
                    self.labels.append(label)

    def predict(self, token):
        # Convert the input token into a feature vector
        input_feature = self.convert_token_to_feature_vector(token)

        # Calculate distances between input feature and all training features
        distances = euclidean_distances([input_feature], self.features)[0]

        # Find the k-nearest neighbors' indices
        k_nearest_indices = np.argsort(distances)[:self.k]

        # Collect the labels of k-nearest neighbors
        k_nearest_labels = [self.labels[i] for i in k_nearest_indices]

        # Predict the most common label among k-nearest neighbors
        predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return predicted_label

    def convert_token_to_feature_vector(self, token):
        # This function should convert a token into a suitable feature vector
        # You need to define how tokens are transformed into features
        pass

def simple_sentiment_analysis(classifier, dev_in_path, dev_out_path):
    output_lines = []
    with open(dev_in_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                token = line
                predicted_label = classifier.predict(token)
                output_lines.append(f"{line} {predicted_label}")
            else:
                output_lines.append("")
    with open(dev_out_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(output_lines))

# Create and train KNN classifiers for different languages
knn_classifier_ES = KNNClassifier(k=5)
knn_classifier_ES.train("Data/ES/train")
simple_sentiment_analysis(knn_classifier_ES, "Data/ES/dev.in", "Data/ES/dev.knn.out")

knn_classifier_RU = KNNClassifier(k=5)
knn_classifier_RU.train("Data/RU/train")
simple_sentiment_analysis(knn_classifier_RU, "Data/RU/dev.in", "Data/RU/dev.knn.out")
