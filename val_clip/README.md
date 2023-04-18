# Image Classification using CLIP

This is a Python program that performs image classification using the CLIP model. The program loads a dataset of images and their categories from a CSV file, preprocesses the images, and feeds them to the CLIP model along with text embeddings corresponding to the category names. The program then evaluates the performance of the model by computing the top-k precision on the dataset.

## Requirements

- Python 3.6 or higher
- PyTorch
- CLIP
- Pandas
- NumPy
- PIL (Python Imaging Library)

## CSV
The CSV file should have the following format:
```
image_name, app name, category
image1.jpg, App1, category1
image2.jpg, App2, category2
...
```
There is a sample CSV file named `app2category.csv` included in this repository.

## Usage

1. Set the desired batch size and top-k value in the `batch_size` and `topk` variables, respectively.
2. Set the paths to the dataset directory and CSV file in the `dataset_dir` and `csv_path` variables, respectively.
3. Run the program using the following command:

```bash
python pred.py --batch-size 1024 --top-k 5 --dataset-dir "/path/to/dataset" --csv-path "/path/to/csv"
```


The program will print the precision of the model after processing each batch of images.

