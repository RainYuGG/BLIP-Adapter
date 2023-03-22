# %%
# Import necessary packages.
import pandas as pd
import os
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from transforms import AutoTokenizer
from PIL import Image

# %%
# set dataset path
screen2words_dir = '/data/screen2words' 
root = '/data/rico/combined'
split_file_path = screen2words_dir + '/split'

# %%
# Transforms
# All we need here is to resize the PIL image and transform it into Tensor.
def tfm(H=256, W=144):
    transform = transforms.Compose([
        # Resize the image into a fixed shape (height = 256, width = 144)
        transforms.Resize((H, W)),
        # transforms.CenterCrop(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    return transform

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

class Screeb2WordsDataset(VisionDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(
        self, 
        root: str, 
        ann_file: str,
        transform: Optional[Callable] = tfm(H=256, W=144),
        target_transform: Optional[Callable] = tokenizer,
        split_file:str,
        split_type='TEST'
    ) -> None:
        """
        Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Annotation Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it. (tokenizer)
        split_file (string): Directory contain how to split.
        split_type: split type, one of 'TRAIN', 'VAL', or 'TEST'
        """
        super(targetDataset).__init__()
        self.root = root
        self.ann_file = ann_file

        assert split_type in {'TRAIN', 'VAL', 'TEST'}
        if split_type == 'TRAIN':
            self.split = [int(line.strip()) for line in open(split_file + '/train_screens.txt', 'r')]
        elif split_type == 'VAL':
            self.split = [int(line.strip()) for line in open(split_file + '/dev.txt', 'r')]
        elif split_type == 'TEST':
            self.split = [int(line.strip()) for line in open(split_file + '/test_screens.txt', 'r')]
        self.data = pd.read_csv(ann_file + '/screen_summaries.csv')
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
        img = Image.open(self.imag_path + '/' + self.data['screenId'][index] + ',jpg')
        if self.transform is not None:
            img = self.transform(img)
        target = self.data['summary'][index]
        if self.target_transform is not None:
            target = self.target_transform(target)  

        return img,target

