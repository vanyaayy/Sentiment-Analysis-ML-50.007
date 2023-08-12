# Sentiment Analysis with Bidirectional LSTM (using PyTorch):


import torch
import numpy as np

# Define LSTM model
class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden_concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(hidden_concat)

# Build vocabulary
def build_vocab(data_path):
    vocab = {}
    word_idx = 0
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, _ = line.strip().rsplit(' ', 1)
            for word in text.split():
                if word not in vocab:
                    vocab[word] = word_idx
                    word_idx += 1
    return vocab

vocab = build_vocab("train.txt")

# Load and preprocess data
def preprocess_data(data_path, vocab):
    texts, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().rsplit(' ', 1)
            text_indices = [vocab[word] for word in text.split()]
            texts.append(text_indices)
            labels.append(int(label))
    return texts, labels

train_texts, train_labels = preprocess_data("train.txt", vocab)
test_texts, test_labels = preprocess_data("test.txt", vocab)

# Define DataLoader
train_features = torch.tensor(train_texts, dtype=torch.long)
train_labels = torch.tensor(train_labels, dtype=torch.long)
train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model training
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = 2  # Binary sentiment classification
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # Adjust number of epochs
    model.train()
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

# Model testing
model.eval()
all_preds = []
with torch.no_grad():
    for batch_features in test_texts:
        batch_features = torch.tensor(batch_features, dtype=torch.long).unsqueeze(0).to(device)
        outputs = model(batch_features)
        preds = np.argmax(outputs.cpu().numpy(), axis=1)
        all_preds.extend(preds)

# Evaluate accuracy
accuracy = np.mean(np.array(all_preds) == np.array(test_labels))
print(f"Accuracy: {accuracy:.2f}")
