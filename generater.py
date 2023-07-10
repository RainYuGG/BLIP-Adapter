#%%
import os
import argparse
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
# own processing implementation
import tfm
from loader import load_model

def generate_caption(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.image_id is not None:
        sampleId = [args.image_id]
    else:
        sampleId = [54137, 13059, 1129, 37226, 12816, 57449]

    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    vis_processors = tfm.tfm()
    image = torch.stack([vis_processors["eval"](Image.open(os.path.join(args.img_dir, str(id) + ".jpg")).convert("RGB")) for id in sampleId])
    image = image.to(device)

    # loads BLIP caption base mode
    model = load_model(args.model, isTrain=False)
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
        print(f"Load checkpoint from {args.checkpoint_path}")
    model.to(device)
    model.eval()



    # generate caption
    caption = model.generate({"image": image})
    print(caption)
    # %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='blip_caption',
                        help='model name')
    parser.add_argument('-ckpt', '--checkpoint_path', type=str, default='./ckpt/b32_e4-15_bleu.ckpt',
                        help='path to model checkpoint')
    parser.add_argument('--img-dir', type=str, default='/data/rico/combined/',
                        help='image directory where Rico dataset is stored')
    parser.add_argument('--image_id', type=int, default=None,
                        help='image id to generate caption')
    args = parser.parse_args()
    generate_caption(args)