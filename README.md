# image captioning based on Screen2Words
image-to-text on Android screenshot

## Datasets.py
 
Implement the VisionDataset.

Load the image from [Rico UI Screenshots and View Hierarchies dataset](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz) and the summary from [Screen2Words dataset](https://github.com/google-research-datasets/screen2words), and deal with the same split within Screen2Words.

#### TODO:

1. ~~vocabulary preprocessing~~ (use tokenizer)
2. ~~model concatenate (ViT encoder& Text Transformer decoder)~~ (use blip)
3. ~~Each image should select once and randomly choose one of the captions. (Instead of five captions and five the same images)~~
4. ~~BLEU score check & how to eval metric and save model~~
   * ~~Due to five captions for an image, may select different captions at each time.~~
   * ~~In validated/test step, Does it need to calculate loss? Or just generates captions and calculates the CIDEr, BLEU score for saving model~~
5. ~~add CIDEr score check [COCOEval](https://blog.csdn.net/weixin_41848012/article/details/121254472)~~
6. batch size issue
   * change input from 384\*384 to 224\*224 and change first conv2d output to fit original featured map size.
   * original image resize 1920\*1080 -> 960\*540
   
   both these implementation don't reduce CUDA memory usage. 
still only can use 32 batch size.

   simplily change input from 384\*384 to 224\*224 can work. 
I think it's due to the intermediate featured map size reducing

   * try to use Gradient Accumulation to get bigger batch size
   
7. ~~learning rate scheduler~~
8. dataset implementation rethink
   * Training Loader problem





## Download repo
```
git clone --recursive https://github.com/RainYuGG/image-captioning-based-on-Screen2Words.git
```

## Install coco-caption
To properly obtain the CIDEr Score in `eval.py`, you need to install the coco_caption package. Follow the steps below to install it:

### 1. Install the model
Clone the `coco_caption` repository and navigate to the cloned directory:
```bash 
cd coco_caption
bash get_stanford_models.sh
pip install gensim
```

### 2. Install java
1. Download the latest version of Java 8 from the Oracle website: https://www.oracle.com/java/technologies/downloads/#java8
2. Copy the downloaded file to your system and extract it with the following commands:
```bash
sudo cp jdk-xxxxx_linux-x64_bin.tar.gz /opt
cd /opt
sudo mkdir java
sudo chown ${USER}:${USER} java
sudo tar -zxvf jdk-xxxxx_linux-x64_bin.tar.gz -C /opt/java
```
3. Set the environment variable in `~/.bashrc` by adding the following lines:
```
#set java environment
export JAVA_HOME=/opt/java/jdk1.8.xx
export PATH=${JAVA_HOME}/bin:${PATH}
```
4. Source the `.bashrc` file to apply the changes:
```bash
source ~/.bashrc
```

5. Verify that Java is installed by checking the version:

```bash
java -version
```

### 3. Install WMD
To install WMD, run the following command:
```bash
bash get_google_word2vec_model.sh
```

### 4. Demo
To run the demo, execute the following command:
```bash
python scorer.py
```
Make sure to run the demo after the installation to confirm that everything works as expected.


## reference

#### dataset

[Rico](https://interactionmining.org/rico)

Screen2Words: [paper](https://arxiv.org/abs/2108.03353) / [code](https://github.com/google-research/google-research/tree/master/screen2words) / [dataset](https://github.com/google-research-datasets/screen2words)

#### some sample code of captioning

https://github.com/yurayli/image-caption-pytorch

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

https://github.com/xiadingZ/video-caption.pytorch

https://github.com/huggingface/transformers/blob/main/examples/pytorch/contrastive-image-text/run_clip.py
