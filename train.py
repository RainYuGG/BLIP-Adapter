#%%
import os
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from lavis.models import load_model_and_preprocess
import numpy as np
import random
import argparse
# This is for the progress bar.
from tqdm.auto import tqdm
# own dataset & scorer utils implement
from s2w_dataset import Screeb2WordsDataset
import scorer

#%%
def main(args):
    # set random args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # "cuda" only when GPUs are available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stale = 0
    best_score = 0.0

    # %%
    model, _ , _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=False, device=device)
    #modelckpt = '/home/chingyu/image-captioning-based-on-Screen2Words/ckpt/b32_e4-15_bleu.ckpt'
    #model.load_state_dict(torch.load(modelckpt))

    # # change input size from 384*384 to 224*224
    # model.visual_encoder.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=(9, 9), stride=(9, 9))
    # model.visual_encoder.patch_embed.img_size = (224, 224)
    model.to(device)

    # %%
    # training preprocessor
    import tfm
    vis_processors = tfm.tfm()

    #%%
    # load dataset
    train_dataset = Screeb2WordsDataset(args.img_dir, args.s2w_dir, 'TRAIN', vis_processors, None, args.caption_type, args.debug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=train_dataset.collate_fn)
    valid_dataset = Screeb2WordsDataset(args.img_dir, args.s2w_dir, 'VALID', vis_processors, None, args.caption_type, args.debug)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=valid_dataset.collate_fn)

    #%%
    # Gradient Accumulation
    bs = args.batch_size * args.accumulation_steps

    # Define the loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(params, lr=args.lr)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr) #, weight_decay=weight_decay)
    num_training_steps = len(train_loader) / args.accumulation_steps * args.num_epochs 
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    #%%
    with open(f"./log/{args.exp_name}_bs{bs}_log.txt","a") as f:
        f.write(f"bs = {bs}({args.batch_size}*{args.accumulation_steps}), num_epoch = {args.num_epochs}\n, lr = {args.lr}, patience = {args.patience}")

    for epoch in range(args.num_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = []
        for index, batch in enumerate(tqdm(train_loader)):
            batch['image'] = batch['image'].to(device)
            # print(batch['image_id'],batch['text_input'])
            batch['text_input'] = batch['text_input']

            # Forward the data. (Make sure data and model are on the same device.)
            output = model(batch)
            loss = output.loss / args.accumulation_steps

            # Bert using gelu and Cross Entropy for loss function automatically.
            # model output contain the Cross Entropy loss.
            # Compute the gradients for parameters.
            loss.backward()

            # Gradient Accumulation
            if (index + 1) % args.accumulation_steps == 0 or (index+1) == len(train_loader):
                # Clip the gradient norms for stable training.
                # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                # Update the parameters with computed gradients.
                optimizer.step()

                # Update the learning rate with scheduler
                scheduler.step()

                # Gradients stored in the parameters in the previous step should be cleared out first.
                optimizer.zero_grad()

            # Record the loss.
            train_loss.append(loss.item())

        train_loss = args.accumulation_steps * sum(train_loss) / len(train_loss)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{args.num_epochs:03d} ] loss = {train_loss:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        caption_predictions = []
        caption_references = []
        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):
            img_input = {"image": batch['image'].to(device)}
            caption_references += batch['text_input']
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                caption_pred = model.generate(img_input)
                caption_predictions += caption_pred
        # valid_score = scorer.calculate_score(caption_predictions, caption_references, 'bleu')
        cocoeval = scorer.Scorers(caption_predictions, caption_references)
        total_score = cocoeval.compute_scores()
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{args.num_epochs:03d} ] bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f}")
        # update logs
        valid_score = total_score['bleu'][3] + total_score['CIDEr']
        if valid_score > best_score:
            with open(f"./log/{args.exp_name}_bs{bs}_log.txt","a") as f:
                f.write(f"[ Valid | {epoch + 1:03d}/{args.num_epochs:03d} ] bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f} -> best\n")
        else:
            with open(f"./log/{args.exp_name}_bs{bs}_log.txt","a") as f:
                f.write(f"[ Valid | {epoch + 1:03d}/{args.num_epochs:03d} ] bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f}\n")
        # save models
        if valid_score > best_score:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"{args.exp_name}_bs{bs}.ckpt") # only save best to prevent output memory exceed error
            best_score = valid_score
            stale = 0
        else:
            stale += 1
            if stale > args.patience:
                print(f"No improvment {args.patience} consecutive epochs, early stopping at {epoch}")
                break

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=0,
                        help='debug mode for testing code')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate during training')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size during training')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='gradient accumulation step')
    parser.add_argument('--patience', type=int, default=15,
                        help='')
    parser.add_argument('--img-dir', type=str, default='/data/rico/combined/',
                        help='image directory where Rico dataset is stored')
    parser.add_argument('--s2w-dir', type=str, default='/data/screen2words/',
                        help='directory where Screen2words dataset is stored')
    parser.add_argument('--caption-type', type=str, default='RANDOM',
                        help='type of select caption in training data.\n \
                            set "RANDOM" to select random one caption for each image in traning.\n \
                            set "FULL" to select all five caption and duplicate image five times for five cases.'
                        )
    parser.add_argument('--exp-name',type=str, default='test',
                        help='experiment name')
    parser.add_argument('--seed', type=int, default=1126,
                        help='set random seed')
    args = parser.parse_args()
    main(args) 