# Sentiment Analysis with BERT (using Hugging Face Transformers):


import torch
import numpy as np

# Define a simple neural network class
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

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

# Define a simple tokenizer (splitting text into individual words)
def simple_tokenizer(text):
    return text.split()

# Convert texts into feature vectors
def convert_texts_to_feature_vectors(texts, tokenizer):
    feature_vectors = []
    for text in texts:
        tokens = tokenizer(text)
        # Convert tokens into a simple average word embedding
        # You need to define how word embeddings are calculated
        # For simplicity, let's assume an average of random numbers for now
        avg_embedding = np.mean(np.random.rand(300))  # Assuming 300-dimensional word embeddings
        feature_vectors.append(avg_embedding)
    return feature_vectors

train_features = convert_texts_to_feature_vectors(train_texts, simple_tokenizer)
test_features = convert_texts_to_feature_vectors(test_texts, simple_tokenizer)

# Convert to PyTorch tensors
train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.int64)
test_features = torch.tensor(test_features, dtype=torch.float32)

# Create DataLoader
train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize and train the simple neural network
input_size = 300  # Assuming 300-dimensional word embeddings
hidden_size = 128
output_size = 2  # Binary classification
model = SimpleNN(input_size, hidden_size, output_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):  # Adjust number of epochs
    model.train()
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

# Testing
model.eval()
all_preds = []
with torch.no_grad():
    for batch_features in test_features.split(16):
        batch_features = batch_features.to(device)
        logits = model(batch_features)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

# Evaluate accuracy
accuracy = np.mean(np.array(all_preds) == np.array(test_labels))
print(f"Accuracy: {accuracy:.2f}")
