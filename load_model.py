#%%
import os
import torch
from PIL import Image
import tfm
import models
import yaml


if __name__ == "__main__":
    raw_image = Image.open("/data/rico/combined/9611.jpg").convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis_processors = tfm.tfm()
    image = vis_processors["train"](raw_image).unsqueeze(0).to(device)
    with open('configs/blip_caption.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = models.make(config['model']).cuda()

    model.load_checkpoint("https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP/blip_coco_caption_base.pth")

    # Print the model architecture
    print(model)

    # Print the model's state_dict() containing the parameters
    print(model.state_dict())
#%%