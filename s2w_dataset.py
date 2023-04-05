# %%
# Import necessary packages.
import pandas as pd
import os
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
        text_processor: Optional[Callable] = None
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
        self.data = self.data[self.data['screenId'].isin(split)].groupby('screenId').agg(list).reset_index(drop=False)#.head(32)
        #tokenizer
        self.text_processor = text_processor
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: dict (image, caption, id).
        """
        img = Image.open(self.img_dir + str(self.data['screenId'][index]) + '.jpg').convert("RGB")
        # if self.split_type == 'TRAIN':
        #     caption = self.data['summary'][index][random.randint(0, 4)]
        # else:
        caption = self.data['summary'][index]
        if self.transform is not None:
            img = self.transform(img)
        if self.text_processor is not None:
            caption = self.text_processor(caption)  

        return {
            "image": img,
            "text_input": caption,
            "image_id": self.data['screenId'][index],
        }
