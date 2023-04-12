# %%
# Import necessary packages.
import pandas as pd
import os
import random
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image


class Screeb2WordsDataset(VisionDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(
        self, 
        img_dir: str, 
        caption_file: str,
        split_dir: str,
        split_type: str = 'TEST', 
        transform: Optional[Callable] = None,
        text_processor: Optional[Callable] = None,
        caption_type: str = None,
        debug: bool = False,
    ) -> None:
        """
        Args:
        img_dir (string): img_dir directory where images are downloaded to.
        caption_file (string): caption Path to caption file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        text_processor (callable, optional): A function/transform that add prompt in the beginnoing of the caption.
        split_dir (string): Directory contain how to split.
        split_type: split type, one of 'TRAIN', 'VAL', or 'TEST'
        """
        super(VisionDataset).__init__()
        self.img_dir = img_dir
        self.caption_file = caption_file
        self.split_type = split_type
        self.caption_type = caption_type
        self.debug = debug
        assert split_type in {'TRAIN', 'VALID', 'TEST'}
        if split_type == 'TRAIN':
            split = [int(line.strip()) for line in open(split_dir + 'train_screens.txt', 'r')]
            self.transform = transform['train']
        elif split_type == 'VALID':
            split = [int(line.strip()) for line in open(split_dir + 'dev_screens.txt', 'r')]
            self.transform = transform['eval']
        elif split_type == 'TEST':
            split = [int(line.strip()) for line in open(split_dir + 'test_screens.txt', 'r')]
            self.transform = transform['eval']
        
        self.data = pd.read_csv(caption_file)
        if self.caption_type == "random":
            self.data = self.data[self.data['screenId'].isin(split)].groupby('screenId').agg(list).reset_index(drop=False)#.head(32)
        elif self.caption_type == "full":
            self.data = self.data[self.data['screenId'].isin(split)].reset_index(drop = True).head(50)

        self.caption_dict = self.data.groupby('screenId')['summary'].apply(list).to_dict
        #tokenizer
        self.text_processor = text_processor
        self.random_number = random.randint(0, 4)
        
    def __len__(self) -> int:
        if self.debug:
            return 4
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: dict (image, caption, id).
        """
        img = Image.open(self.img_dir + str(self.data['screenId'][index]) + '.jpg').convert("RGB")
        caption = self.data['summary'][index]
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "text_input": caption,
            "image_id": self.data['screenId'][index],
        }

    def collate_fn(self, samples):
        
        image_batch = [x["image"] for x in samples]
        caption_batch = [x["text_input"] for x in samples]
        id_batch = [x["image_id"] for x in samples]

        if self.caption_type == "random" and self.split_type == "TRAIN":
            caption_batch = [x["text_input"][random.randint(0, 4)] for x in samples]

        if self.text_processor is not None:
            caption_batch = [self.text_processor(x) for x in caption_batch]

        return{
            "image" : image_batch,
            "text_input" : caption_batch,
            "image_id" : id_batch,
        }

    def update_caption(self):
        if self.caption_type == "random":
            self.random_number = random.randint(0, 4)
