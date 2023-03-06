import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from image_captioning_dataset import CaptioningDataset
from vision_transformer import ViT
from text_transformer import Transformer

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set hyperparameters
embed_size = 512
hidden_size = 512
num_layers = 1
num_heads = 8
num_epochs = 10
batch_size = 128
learning_rate = 0.0001

# load dataset
data_folder = "/path/to/flickr30k/folder/"
caption_file = os.path.join(data_folder, "captions.txt")
vocab_file = os.path.join(data_folder, "vocab.pkl")
image_folder = os.path.join(data_folder, "images")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
dataset = CaptioningDataset(caption_file, vocab_file, image_folder, transform)
vocab_size = len(dataset.vocab)

# split dataset into training and validation set
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# define dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# initialize model
encoder = ViT().to(device)
decoder = Transformer(embed_size, hidden_size, vocab_size, num_layers, num_heads).to(device)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab["<pad>"])
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

# train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)

        # forward pass
        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])

        # compute loss
        loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training status
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # evaluate the model on the validation set
    with torch.no_grad():
        total_loss = 0
        total_words = 0
        for images, captions in val_loader:
            images = images.to(device)
            captions = captions.to(device)
            features = encoder(images)
            outputs = decoder.generate(features, dataset.vocab["<start>"], dataset.vocab["<end>"])
            targets = captions[:, 1:]
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-

