import argparse
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class ImageFeatureDataset(Dataset):
    def __init__(self, df, dataset_dir):
        self.dataset_dir = dataset_dir
        self.df = df
        self.image_name = df["image_name"].tolist()
        self.catagorys = [x[1:] for x in df["category"].tolist()]
        self.unique_category = list(set(self.catagorys))
        # create a one hot encoder in dataset
        self.onehot = [self.category_to_onehot(x) for x in self.catagorys]
        # preprocess image

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        if self.dataset_dir[-1] != "/":
            self.dataset_dir += "/"
        path = self.dataset_dir + self.image_name[idx]
        ground_truth = self.onehot[idx]
        # print(type(ground_truth), type(path))
        return path, ground_truth
    
    def category_to_onehot(self, category):
        index = self.unique_category.index(category)
        onehot = [0] * len(self.unique_category)
        onehot[index] = 1
        return onehot

    def get_category(self):
        return self.unique_category
    
    def colate_fn(self, samples):
        images = [preprocess(Image.open(sample[0])).unsqueeze(0).to(device) for sample in samples]
        images = torch.cat(images, dim=0)
        ground_truth = [sample[1] for sample in samples]
        ground_truth = torch.tensor(ground_truth)

        return images, ground_truth

def train(batch_size, topk, dataset_dir, csv_path):
    df = pd.read_csv(csv_path)
    
    image_dataset = ImageFeatureDataset(df, dataset_dir)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, collate_fn=image_dataset.colate_fn)

    unique_category = image_dataset.get_category()
    print("Unique catagory: ", unique_category)

    texts = torch.cat([clip.tokenize(f"a screen shot of {c} app") for c in unique_category]).to(device)
    total_percision = 0
    #start evaluation and get percision
    for i, (images, ground_truth) in enumerate(image_dataloader):
        
        print(f"batch {i} of {len(image_dataloader)}")
        
        with torch.no_grad():
            logits_per_image, logits_per_text = model(images, texts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        probs = [np.argpartition(p, -topk)[-topk:] for p in probs]

        total_percision += sum(1 for x, y in zip(probs, ground_truth.argmax(axis=1)) if y.item() in x)
        print(f"Precision: {total_percision/((i+1)*batch_size)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image classification model using CLIP")
    parser.add_argument("--batch-size", type=int, default=2048, help="batch size for training")
    parser.add_argument("--top-k", type=int, default=5, help="top-k precision for evaluation")
    parser.add_argument("--dataset-dir", type=str, required=True, help="path to the dataset directory")
    parser.add_argument("--csv-path", type=str, required=True, help="path to the CSV file containing image names and categories")

    args = parser.parse_args()

    train(args.batch_size, args.top_k, args.dataset_dir, args.csv_path)
