#%%
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, AutoTokenizer
from lavis.models import load_model_and_preprocess
# This is for the progress bar.
from tqdm.auto import tqdm
# ViT & Transformer
# from vision_transformer import ViT
# from text_transformer import Transformer

# own dataset implement
from s2w_dataset import Screeb2WordsDataset
import scorer

DEBUG = False

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
batch_size = 128

# initialize model & tokenizer define
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
modelckpt = '/home/chingyu/image-captioning-based-on-Screen2Words/ckpt/b32_e4-15_bleu.ckpt'
model.load_state_dict(torch.load(modelckpt))
model.to(device)

#%%
# load dataset
test_dataset = Screeb2WordsDataset(img_dir, caption_file, split_dir, 'TEST', vis_processors, None)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model.eval()
caption_predictions = []
caption_references = []
for batch in tqdm(test_loader):
    img_input = {"image": batch['image'].to(device)}
    caption_references += scorer.transpose(batch['text_input'])
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        caption_pred = model.generate(img_input)
        caption_predictions += caption_pred
    
    # Only run one epoch if DEBUG is true
    if DEBUG:
        break
#%%
print('ref:', len(caption_references))
print('ref:', len(caption_references[0]))
print('pred:', len(caption_predictions))
# print('id:', batch['image_id'])
#%%    
res = scorer.calculate_score(caption_predictions, caption_references, 'bleu')
print(res)

res = scorer.calculate_score(caption_predictions, caption_references, 'rouge')
print(res)

res = scorer.calculate_score(caption_predictions, caption_references, 'meteor')
print(res)

# Add scorer to calculate the CIDEr and other scores.
scorer = scorer.Scorers(caption_predictions, caption_references)
res = scorer.compute_scores()
print(res)
