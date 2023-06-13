#%%
import os
import requests
import yaml
import torch
import torch.nn as nn
from typing import Union, List
from PIL import Image
import models
from transformers.adapters import LoRAConfig, AdapterConfig

def load_model(model_name: str, isTrain: bool = False):
    with open(os.path.join('configs/', model_name + '.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = models.make(config['model']).cuda()
    if(model_name == 'blip_caption'):
        model.load_checkpoint(config['model']['checkpoint_url'])

    trainable_name = []

    if 'adapter_type' in config['model']['args']:
        trainable_name.append('prompt')

    # load adapter to bert
    if 'bert_adapter' in config['model']:
        bert_adapter = config['model']['bert_adapter']
        trainable_name.append(bert_adapter)
        load_adapter(model, bert_adapter)
        if isTrain:
            model.text_decoder.train_adapter(bert_adapter)

    if 'tune_language' in config['model'] and config['model']['tune_language']:
        trainable_name.append('text_decoder')
    
    if len(trainable_name) != 0:
        freeze_parameters(model, trainable_name)

    print_trainable_parameters(model, True)

    print("-" * 20)

    return model

def load_adapter(model: nn.Module, bert_adapter: str):
    # load adapter config for language model
    if bert_adapter == "bottleneck_adapter":
        config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
        model.text_decoder.add_adapter(bert_adapter, config=config)
    #load LoRA config for language model
    elif bert_adapter == "lora_adapter":
        config = LoRAConfig(r=8, alpha=16)
        model.text_decoder.add_adapter(bert_adapter, config=config)


def freeze_parameters(model: nn.Module, trainable_name: Union[str, List[str]]):
    if isinstance(trainable_name, str):
        trainable_name = [trainable_name]
    for name, param in model.named_parameters():
        # train either vit adapter or language model
        if any(n in name for n in trainable_name):
            continue
            # print(name)
        else:
            param.requires_grad_(False)


def print_trainable_parameters(model: nn.Module, show_trained_param: bool = False):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if show_trained_param:
                print(name)
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
    # Print the model architecture
    # print(model)

    # Print the model's state_dict() containing the parameters
    # print(model.state_dict())

    # generate caption
    caption = model.generate({"image": image})
    print(caption)