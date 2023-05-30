#%%
import torch
from torch.utils.data import DataLoader
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
import tfm
from loader import load_model

def evaluation(args):
    # Set deterministic options for reproducible results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # "cuda" only when GPUs are available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model & tokenizer define
    model = load_model(args.model)
    # load checkpoint
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
        print(f"Load checkpoint from {args.checkpoint_path}")

    # training preprocessor
    import tfm
    vis_processors = tfm.tfm()
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
    if args.debug:
        print('ref len:', len(caption_references))
        print('ref[0] len:', len(caption_references[0]))
        print('pred len:', len(caption_predictions))
    # print('id:', batch['image_id'])
    #%%    
    bleu = scorer.calculate_score(caption_predictions, caption_references, 'bleu')
    print(f'bleu:{bleu:.5f}')

    rougeL = scorer.calculate_score(caption_predictions, caption_references, 'rouge')
    print(f'rougeL:{rougeL:.5f}')

    mentor = scorer.calculate_score(caption_predictions, caption_references, 'meteor')
    print(f'mentor:{mentor:.5f}')

    # Add scorer to calculate the CIDEr and other scores.
    cocoeval = scorer.Scorers(caption_predictions, caption_references)
    total_score = cocoeval.compute_scores()
    print(f"bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=0,
                        help='debug mode for testing code')
    parser.add_argument('-m', '--model', type=str, default='blip_caption',
                        help='model name')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='batch size during training')
    parser.add_argument('--img-dir', type=str, default='/data/rico/combined/',
                        help='image directory where Rico dataset is stored')
    parser.add_argument('--s2w-dir', type=str, default='/data/screen2words/',
                        help='directory where Screen2words dataset is stored')
    parser.add_argument('-c', '--caption-type', type=str, default='EVAL',
                        help='type of select caption in training data.\n \
                            set "RANDOM" to select random one caption for each image in traning.\n \
                            set "FULL" to select all five caption and duplicate image five times for five cases.'
                        )
    parser.add_argument('-ckpt', '--checkpoint_path', type=str, default='./ckpt/b32_e4-15_bleu.ckpt',
                        help='path to model checkpoint')
    args = parser.parse_args()
    evaluation(args) 