# Sentiment Analysis with BERT (using Hugging Face Transformers):


import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary sentiment

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

# Tokenize and create DataLoader
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(5):  # Adjust number of epochs
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Testing
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = np.argmax(logits.cpu().numpy(), axis=1)
        all_preds.extend(preds)

# Evaluate accuracy
accuracy = np.mean(np.array(all_preds) == np.array(test_labels))
print(f"Accuracy: {accuracy:.2f}")
