# %%
# Import necessary packages.
import os
import random
import torch
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import polars as pl


class Screeb2WordsDataset(VisionDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(
        self, 
        img_dir: str, 
        s2w_dir: str,
        split_type: str = 'TEST', 
        transform: Optional[Callable] = None,
        text_processor: Optional[Callable] = None,
        caption_type: str = 'EVAL',
        debug: bool = False,
    ) -> None:
        """
        Args:
        img_dir (string): directory where images are downloaded to.
        s2w_dir (string): directory where Screen2words dataset is stored.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        text_processor (callable, optional): A function/transform that add prompt in the beginnoing of the caption.
        split_dir (string): Directory contain how to split.
        split_type: split type, one of 'TRAIN', 'VAL', or 'TEST'
        """
        super(VisionDataset).__init__()
        # set data path
        self.img_dir = img_dir
        caption_file = os.path.join(s2w_dir + 'screen_summaries.csv')
        split_dir = os.path.join(s2w_dir + 'split')
        assert split_type in {'TRAIN', 'VALID', 'TEST'}
        self.split_type = split_type
        if split_type == 'TRAIN':
            split = [int(line.strip()) for line in open(os.path.join(split_dir, 'train_screens.txt'), 'r')]
            self.transform = transform['train']
        elif split_type == 'VALID':
            split = [int(line.strip()) for line in open(os.path.join(split_dir, 'dev_screens.txt'), 'r')]
            self.transform = transform['eval']
        elif split_type == 'TEST':
            split = [int(line.strip()) for line in open(os.path.join(split_dir, 'test_screens.txt'), 'r')]
            self.transform = transform['eval']

        assert caption_type in {'RANDOM', 'FULL', 'EVAL'}
        self.caption_type = caption_type
        self.data = pl.read_csv(caption_file)
        if self.caption_type == 'FULL' and self.split_type == 'TRAIN':
            self.data = self.data.filter(self.data['screenId'].is_in(split)).sort('screenId')
        else:
            self.data = self.data.filter(self.data['screenId'].is_in(split)).sort('screenId').groupby('screenId').agg(pl.col('*').alias('summary')).sort('screenId')

        if debug:
            self.data = self.data.head(16)
        #tokenizer
        self.text_processor = text_processor
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {image, caption, id}
        """
        img = Image.open(os.path.join(self.img_dir, str(self.data['screenId'][index]) + '.jpg')).convert('RGB')
        caption = self.data['summary'][index]
        if not(self.caption_type == 'FULL' and self.split_type == 'TRAIN'):
            caption = list(caption)
        if self.transform is not None:
            img = self.transform(img)
        return {
            'image': img,
            'text_input': caption,
            'image_id': self.data['screenId'][index],
        }

    def collate_fn(self, samples):
        """
        samples: 
            batch from __getitem__
        Returns: 
            dict {image (tensor), caption (string), id (int)}
        """
        image_batch = torch.stack([x['image'] for x in samples])
        caption_batch = [x['text_input'] for x in samples]
        id_batch = [x['image_id'] for x in samples]

        if self.caption_type == 'RANDOM' and self.split_type == 'TRAIN':
            caption_batch = [x['text_input'][random.randint(0, 4)] for x in samples]

        if self.text_processor is not None:
            caption_batch = [self.text_processor(x) for x in caption_batch]

        return{
            'image' : image_batch,
            'text_input' : caption_batch,
            'image_id' : id_batch,
        }
