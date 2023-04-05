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
# own dataset & score utils implement
from s2w_dataset import Screeb2WordsDataset
import score

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
num_epochs = 100
patience = 10
batch_size = 32
learning_rate = 1e-4
weight_decay = 0.05
# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_bleu = 0.0
_exp_name = "bleu"

# %%
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=False, device=device)

# load dataset
train_dataset = Screeb2WordsDataset(img_dir, caption_file, split_dir, 'TRAIN', vis_processors, None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_dataset = Screeb2WordsDataset(img_dir, caption_file, split_dir, 'VALID', vis_processors, None)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


#%%

# Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(params, lr=learning_rate)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate) #, weight_decay=weight_decay)

#%%
for epoch in range(num_epochs):
    # ---------- Training ----------
    model.train()
    train_loss = []
    for batch in tqdm(train_loader):
        batch['image'] = batch['image'].to(device)
        # print(batch['image_id'],batch['text_input'])
        batch['text_input'] = batch['text_input'][random.randint(0,4)]
        # Forward the data. (Make sure data and model are on the same device.)
        output = model(batch)
        # bert using gelu and Cross Entropy for loss function automatically.
        # model output contain the cross-entropy loss.
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
        
        # Record the loss and accuracy.
        train_loss.append(loss.item())
        
    train_loss = sum(train_loss) / len(train_loss)
    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    caption_predictions = []
    caption_references = []
    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):
        img_input = {"image": batch['image'].to(device)}
        caption_references += score.transpose(batch['text_input'])
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            caption_pred = model.generate(img_input)
            caption_predictions += caption_pred
    valid_bleu = score.calculate_bleu(caption_predictions, caption_references)
    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] bleu = {valid_bleu:.5f}")
    # update logs
    if valid_bleu > best_bleu:
        with open(f"./{_exp_name}_log.txt","a") as f:
            f.write(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] bleu = {valid_bleu:.5f} -> best\n")
    else:
        with open(f"./{_exp_name}_log.txt","a") as f:
            f.write(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] bleu = {valid_bleu:.5f}\n")
    # save models
    if valid_bleu > best_bleu:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"b{batch_size}_e{num_epochs}_{_exp_name}.ckpt") # only save best to prevent output memory exceed error
        best_bleu = valid_bleu
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping at {epoch}")
            break

#%%