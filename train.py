#%%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, AutoTokenizer
from lavis.models import load_model_and_preprocess
import random
# This is for the progress bar.
from tqdm.auto import tqdm
# ViT & Transformer
# from vision_transformer import ViT
# from text_transformer import Transformer

# own dataset implement
from datasets import Screeb2WordsDataset
#%%
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# "cuda" only when GPUs are available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% 
# set data path
img_dir = '/data/rico/combined/'
screen2words_dir = '/data/screen2words/'
caption_file = screen2words_dir + '/screen_summaries.csv'
split_dir = screen2words_dir + 'split/'

# set hyperparameters
# parameter for training 
num_epochs = 20
patience = 3
batch_size = 32
learning_rate = 0.0001
weight_decay = 0.05

# %%
# Transforms
# All we need here is to resize the PIL image and transform it into Tensor.
# def tfm(H=256, W=144):
#     transform = transforms.Compose([
#         # Resize the image into a fixed shape (height = 256, width = 144)
#         transforms.Resize((H, W)),
#         # transforms.CenterCrop(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
#     ])
#     return transform


# initialize model & tokenizer define
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=False, device=device)

# load dataset
train_dataset = Screeb2WordsDataset(img_dir, caption_file, split_dir, 'TRAIN', vis_processors, None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# valid_dataset = Screeb2WordsDataset(img_dir, caption_file, split_dir, 'VALID', vis_processors, None)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


#%%

# Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# params = list(encoder.parameters()) + list(decoder.parameters())
# optimizer = optim.Adam(params, lr=learning_rate)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate) #, weight_decay=weight_decay)

# Initialize trackers, these are not parameters and should not be changed
stale = 0
# best_acc = 0
_exp_name = "no_metric"
#%%
for epoch in range(num_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        batch['image'] = batch['image'].to(device)
        # print(batch['image_id'],batch['text_input'])
        batch['text_input'] = batch['text_input'][random.randint(0,4)]

        # Forward the data. (Make sure data and model are on the same device.)
        output = model(batch)

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        # bert using gelu and Cross Entropy for loss function automatically.
        loss = output.loss
        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()
        
        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update the parameters with computed gradients.
        optimizer.step()

        # TODO check BLEU or others score
        # Compute the accuracy for current batch.
        # acc = (logits.argmax(dim=-1) == caption.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        # train_accs.append(acc)
        
    train_loss = sum(train_loss) / len(train_loss)
    # train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    # valid_loss = []
    # valid_accs = []

    # # Iterate the validation set by batches.
    # for batch in tqdm(valid_loader):
    #     batch['image'] = batch['image'].to(device)
    #     # We don't need gradient in validation.
    #     # Using torch.no_grad() accelerates the forward process.
    #     with torch.no_grad():
    #         # logits = model(imgs.to(device))
    #         output = model(batch)

    #     # We can still compute the loss (but not the gradient).
    #     loss = output.loss

    #     # Compute the accuracy for current batch.
    #     # acc = (logits.argmax(dim=-1) == caption.to(device)).float().mean()

    #     # Record the loss and accuracy.
    #     valid_loss.append(loss.item())
    #     # valid_accs.append(acc)
    #     #break

    # # The average loss and accuracy for entire validation set is the average of the recorded values.
    # valid_loss = sum(valid_loss) / len(valid_loss)
    # # valid_acc = sum(valid_accs) / len(valid_accs)

    # # Print the information.
    # print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}")

    # # update logs
    # with open(f"./{_exp_name}_log.txt","a"):
    #         print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}")

    # save models
if True: #valid_acc > best_acc:
    print(f"Best model found at epoch {num_epochs}, saving model")
    torch.save(model.state_dict(), f"b{batch_size}_e{num_epochs}_{_exp_name}.ckpt") # only save best to prevent output memory exceed error
    best_acc = 1#valid_acc
    stale = 0
    # else:
    #     stale += 1
    #     if stale > patience:
    #         print(f"No improvment {patience} consecutive epochs, early stopping")
    #         break

#%%