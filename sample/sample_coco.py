import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from transformers import DistilBertTokenizer
#from vision_transformer_pytorch import VisionTransformer
from image_captioning_dataset import ImageCaptioningDataset

# Define the model
encoder = VisionTransformer(image_size=224, patch_size=16, num_classes=0, dim=768, depth=12, heads=12, mlp_dim=3072)
decoder = TextTransformerDecoder(vocab_size=30522)
model = ImageToTextTransformer(encoder, decoder)

# Define the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define the data loaders
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
train_dataset = ImageCaptioningDataset(CocoCaptions(root='path/to/coco/train2017', annFile='path/to/coco/annotations/captions_train2017.json'), tokenizer, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = ImageCaptioningDataset(CocoCaptions(root='path/to/coco/val2017', annFile='path/to/coco/annotations/captions_val2017.json'), tokenizer, transform)
val_loader = DataLoader(val_dataset, batch_size=16)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
num_epochs = 2
for epoch in range(num_epochs):
    # Train for one epoch
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    # Evaluate on the validation set
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    # Print the results
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

