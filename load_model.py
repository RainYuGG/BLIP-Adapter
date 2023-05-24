#%%
import requests
import yaml
import torch
from PIL import Image

import tfm
import models


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    raw_image = Image.open("/data/rico/combined/9611.jpg").convert("RGB")
    vis_processors = tfm.tfm()
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    with open('configs/blip_caption.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = models.make(config['model']).cuda()
    model.load_checkpoint("https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP/blip_coco_caption_base.pth")

    # Print the model architecture
    # print(model)

    # Print the model's state_dict() containing the parameters
    # print(model.state_dict())

    # generate caption
    caption = model.generate({"image": image})
    print(caption)
#%%