#%%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, AutoTokenizer
# ViT & Transformer
# from vision_transformer import ViT
# from text_transformer import Transformer

# own dataset implement
from datasets import Screeb2WordsDataset
#%%
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#%% 
img_path = '/data/rico/combined/'
screen2words_dir = '/data/screen2words/'
anntation_path = screen2words_dir + '/screen_summaries.csv'
split_dir = screen2words_dir + 'split/'

# TODO
# vocab_file = os.path.join(data_folder, "vocab.pkl")

# set hyperparameters
embed_size = 512
hidden_size = 512
num_layers = 1
num_heads = 8
num_epochs = 10
batch_size = 128
learning_rate = 0.0001

# %%
# Transforms
# All we need here is to resize the PIL image and transform it into Tensor.
def tfm(H=256, W=144):
    transform = transforms.Compose([
        # Resize the image into a fixed shape (height = 256, width = 144)
        transforms.Resize((H, W)),
        # transforms.CenterCrop(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    return transform
#tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# load dataset
train_dataset = Screeb2WordsDataset(img_path, anntation_path, split_dir, 'TRAIN', tfm(224, 224), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = Screeb2WordsDataset(img_path, anntation_path, split_dir, 'VAL', tfm(224, 224), tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16)

# TODO
# initialize model
# encoder = ViT().to(device)
# decoder = Transformer(embed_size, hidden_size, vocab_size, num_layers, num_heads).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=learning_rate)


num_epochs = 2
for epoch in range(num_epochs):
    # Train for one epoch
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    # Evaluate on the validation set
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    # Print the results
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# %%
