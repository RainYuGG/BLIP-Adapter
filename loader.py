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
    if 'adapter_type' in config['model']['args'] or config['model']['tune_args']['tune_language']:
        for name, param in model.named_parameters():
            # train either vit adapter or language model
            if ('adapter_type' in config['model']['args'] and "prompt" in name) or \
                (config['model']['tune_args']['tune_language'] and "text" in name):
                print(name)
            else:
                param.requires_grad_(False)


    return model

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    raw_image = Image.open("/data/rico/combined/9611.jpg").convert("RGB")

    import tfm
    vis_processors = tfm.tfm()
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    model = load_model('blip_caption')
    print_trainable_parameters(model)
    # Print the model architecture
    # print(model)

    # Print the model's state_dict() containing the parameters
    # print(model.state_dict())

    # generate caption
    caption = model.generate({"image": image})
    print(caption)
#%%
