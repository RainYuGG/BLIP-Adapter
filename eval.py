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
import argparse
# This is for the progress bar.
from tqdm.auto import tqdm
# ViT & Transformer
# from vision_transformer import ViT
# from text_transformer import Transformer

# own dataset implement
from s2w_dataset import Screeb2WordsDataset
import scorer

def main(args):

    #%%
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # "cuda" only when GPUs are available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #%% 
    # initialize model & tokenizer define
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)

    #%%
    # load dataset
    test_dataset = Screeb2WordsDataset(args.img_dir, args.s2w_dir, 'TEST', vis_processors, None, args.caption_type, args.debug)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=test_dataset.collate_fn)

    model.eval()
    caption_predictions = []
    caption_references = []
    for batch in tqdm(test_loader):
        img_input = {"image": batch['image'].to(device)}
        caption_references += batch['text_input']
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            caption_pred = model.generate(img_input)
            caption_predictions += caption_pred
        
        # Only run one epoch if DEBUG is true
        if args.debug:
            break
    #%%
    print('ref len:', len(caption_references))
    print('ref[0] len:', len(caption_references[0]))
    print('pred len:', len(caption_predictions))
    # print('id:', batch['image_id'])
    #%%    
    res = scorer.calculate_score(caption_predictions, caption_references, 'bleu')
    print('bleu:', res)

    res = scorer.calculate_score(caption_predictions, caption_references, 'rouge')
    print('rouge:', res)

    res = scorer.calculate_score(caption_predictions, caption_references, 'meteor')
    print('mentor:', res)

    # Add scorer to calculate the CIDEr and other scores.
    cocoeval = scorer.Scorers(caption_predictions, caption_references)
    res = cocoeval.compute_scores()
    print(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=0,
                        help='debug mode for testing code')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size during training')
    parser.add_argument('--img-dir', type=str, default='/data/rico/combined/',
                        help='image directory where Rico dataset is stored')
    parser.add_argument('--s2w-dir', type=str, default='/data/screen2words/',
                        help='directory where Screen2words dataset is stored')
    parser.add_argument('--caption-type', type=str, default='',
                        help='type of select caption in training data.\n \
                            set "RANDOM" to select random one caption for each image in traning.\n \
                            set "FULL" to select all five caption and duplicate image five times for five cases.'
                        )
    parser.add_argument('--ckpt', type=str, default='./ckpt/b32_e4-15_bleu.ckpt',
                        help='path to model checkpoint')
    args = parser.parse_args()
    main(args) 