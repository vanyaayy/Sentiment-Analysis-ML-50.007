# Sentiment Analysis with Bidirectional LSTM (using PyTorch):


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden_concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(hidden_concat)
    
def build_vocab(data_path):
    vocab = Counter()
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, _ = line.strip().rsplit(' ', 1)
            vocab.update(text.split())
    return {word: idx for idx, (word, _) in enumerate(vocab.items())}

vocab = build_vocab("train.txt")

# Load and preprocess data
def preprocess_data(data_path, vocab):
    texts, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().rsplit(' ', 1)
            texts.append([vocab[word] for word in text.split()])
            labels.append(int(label))
    return texts, labels

train_texts, train_labels = preprocess_data("train.txt", vocab)
test_texts, test_labels = preprocess_data("test.txt", vocab)

# Create DataLoader
train_dataset = TensorDataset(torch.tensor(train_texts), torch.tensor(train_labels))
test_dataset = TensorDataset(torch.tensor(test_texts), torch.tensor(test_labels))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Training
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = 2  # Binary sentiment classification
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # Adjust number of epochs
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Testing
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        outputs = model(inputs)
        preds = np.argmax(outputs.cpu().numpy(), axis=1)
        all_preds.extend(preds)

# Evaluate accuracy
accuracy = np.mean(np.array(all_preds) == np.array(test_labels))
print(f"Accuracy: {accuracy:.2f}")
