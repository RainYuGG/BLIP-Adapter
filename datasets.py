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
        root: str, 
        ann_file: str,
        split_dir: str,
        split_type: str = 'TEST', 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        """
        Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Annotation Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it. (tokenizer)
        split_dir (string): Directory contain how to split.
        split_type: split type, one of 'TRAIN', 'VAL', or 'TEST'
        """
        super(VisionDataset).__init__()
        self.root = root
        self.ann_file = ann_file

        assert split_type in {'TRAIN', 'VALID', 'TEST'}
        if split_type == 'TRAIN':
            split = [int(line.strip()) for line in open(split_dir + 'train_screens.txt', 'r')]
        elif split_type == 'VALID':
            split = [int(line.strip()) for line in open(split_dir + 'dev_screens.txt', 'r')]
        elif split_type == 'TEST':
            split = [int(line.strip()) for line in open(split_dir + 'test_screens.txt', 'r')]
        self.data = pd.read_csv(ann_file)
        self.data = self.data[self.data['screenId'].isin(split)]
        self.transform = transform
        #tokenizer
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of targets for the image.
        """
        img = Image.open(self.imag_path + self.data['screenId'][index] + '.jpg')
        if self.transform is not None:
            img = self.transform(img)
        target = self.data['summary'][index]
        if self.target_transform is not None:
            target = self.target_transform(target)  

        return img,target