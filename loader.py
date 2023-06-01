#%%
import os
import requests
import yaml
import torch
from typing import Optional, Dict, Any
from PIL import Image
import models


def load_model(model_name: str):
    with open(os.path.join('configs/', model_name + '.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = models.make(config['model']).cuda()
    if(model_name == 'blip_caption'):
        model.load_checkpoint(config['model']['checkpoint_url'])

    # freeze all parameters except prompt and language model
    if config['model']['args']['adapter_type'] is not None:
        for name, param in model.named_parameters():
            if "prompt" not in name and "text" not in name:
                param.requires_grad_(False)
            else:
                print(name)
                
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    raw_image = Image.open("/data/rico/combined/9611.jpg").convert("RGB")

    import tfm
    vis_processors = tfm.tfm()
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    model = load_model('blip_caption')

    # Print the model architecture
    # print(model)

    # Print the model's state_dict() containing the parameters
    # print(model.state_dict())

    # generate caption
    caption = model.generate({"image": image})
    print(caption)
#%%