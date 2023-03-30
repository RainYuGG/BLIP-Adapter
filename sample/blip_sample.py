#%%
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

raw_image = Image.open("/data/rico/combined/9611.jpg").convert("RGB")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, text_processors = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=False, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
# due to only one img, need to unsqueeze to get B domain.
image = vis_processors["train"](raw_image).unsqueeze(0).to(device)
#%%
print(text_processors["train"])
raw_text = "a screenshot of\n"
print(raw_text)
print(repr(raw_text))
text_input = text_processors["train"](raw_text)
print(repr(text_input))
#%%
samples = {"image": image, "text_input": text_input}
output = model(samples)
print(output.keys())
print(output.loss)
print(output.loss.item())
print(output.intermediate_output.image_embeds.shape)
print(output.intermediate_output.decoder_labels.shape)
# generate caption
caption = model.generate({"image": image, "text_input": text_input})
print(caption)
# %%
