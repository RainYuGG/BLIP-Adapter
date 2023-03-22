# s2w
image-to-text on Android screenshot

## Datasets.py
 
Implement the VisionDataset.

Load the image from [Rico UI Screenshots and View Hierarchies dataset](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz) and the summary from [Screen2Words dataset](https://github.com/google-research-datasets/screen2words), and deal with the same split within Screen2Words.


#### TODO:

1. ~~vocabulary preprocessing~~ (use tokenizer)
2. model concatenate (ViT encoder& Text Transformer decoder)
3. BLEU score check


## reference

#### dataset

[Rico](https://interactionmining.org/rico)

Screen2Words: [paper](https://arxiv.org/abs/2108.03353) / [code](https://github.com/google-research/google-research/tree/master/screen2words) / [dataset](https://github.com/google-research-datasets/screen2words)

#### some sample code of captioning

https://github.com/yurayli/image-caption-pytorch

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

https://github.com/xiadingZ/video-caption.pytorch

https://github.com/huggingface/transformers/blob/main/examples/pytorch/contrastive-image-text/run_clip.py
