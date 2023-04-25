#%%
import os
import argparse
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
# own processing implementation
import tfm

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
    model, _ , _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)
    model.eval()



    # generate caption
    caption = model.generate({"image": image})
    print(caption)
    # %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./ckpt/64/b+c_bs64_FULL.ckpt',
                        help='path to model checkpoint')
    parser.add_argument('--img-dir', type=str, default='/data/rico/combined/',
                        help='image directory where Rico dataset is stored')
    parser.add_argument('--image_id', type=int, default=None,
                        help='image id to generate caption')
    args = parser.parse_args()
    generate_caption(args)