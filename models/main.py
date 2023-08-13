import torch
import torch.nn as nn
import torch.optim as optim
import json

with open('dataset.json', 'r') as json_file:
    data = json.load(json_file)

original_texts = [item['original'] for item in data]
hashed_texts = [item['hashed'] for item in data]

vocab = set()
for text in original_texts + hashed_texts:
    vocab.update(text.split())

vocab = sorted(vocab)
word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for idx, word in enumerate(vocab)}
vocab_size = len(vocab)


input_data = [torch.tensor([word_to_index[word] for word in text.split()], dtype=torch.long) for text in original_texts]
output_data = [torch.tensor([word_to_index[word] for word in text.split()], dtype=torch.long) for text in hashed_texts]

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output
    
d_model = 256
nhead = 8
num_encoder_layers = 4
num_decoder_layers = 4
model = Transformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    for src, tgt in zip(input_data, output_data):
        optimizer.zero_grad()
        output = model(src.unsqueeze(1), tgt.unsqueeze(1)[:-1])
        loss = criterion(output.view(-1, vocab_size), tgt[1:])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(input_data):.4f}')