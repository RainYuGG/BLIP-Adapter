# image captioning based on Screen2Words

## Download repo
```
git clone --recursive https://github.com/RainYuGG/image-captioning-based-on-Screen2Words.git
```

## Build Environment & Requirement

* Use conda to build the environment
```
conda env create -f environment.yml
```

* Install [coco-caption](#install-coco-caption) for evaluation (BLEU, CIDEr).

## Config file

Here are some arguments of partial tuning from the configuration file that are used to build the BLIP adapter model.

* ```adapter_type: "vit"``` : vit adapter ("vit", "vit_grayscale")
     
* ```bert_adapter: "bottleneck_adapter"``` : bert adapter ("bottleneck_adapter", "lora_adapter", or other implementations in adapterhub)

    * if you want to use other implementations in adapterhub, you need to simply modify the code in ```loader.py```
  
* ```tune_language: false``` : tune whole language model or not (True or False)


## Train

```
python train.py --img-dir /path/to/rico --s2w-dir /path/to/screen2words -e 30 -b 32 -p 15
```

## Evaluation

```
python eval.py -ckpt /path/to/checkpoint
```

## Generate Caption

```
python generater.py --img-dir /path/to/rico python train.py -ckpt /path/to/checkpoint --image_id 54137
```

##

### Install coco-caption
To properly obtain the CIDEr Score in `eval.py`, you need to install the coco_caption package. Follow the steps below to install it:

#### 1. Install the model
Clone the `coco_caption` repository and navigate to the cloned directory:
```bash 
cd coco_caption
bash get_stanford_models.sh
pip install gensim
```

#### 2. Install java
1. Download the latest version of Java 8 from the Oracle website: [Java 8](https://www.oracle.com/java/technologies/downloads/#java8)
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

#### 3. Install WMD
To install WMD, run the following command:
```bash
bash get_google_word2vec_model.sh
```

#### 4. Demo
To run the demo, execute the following command:
```bash
python scorer.py
```
Make sure to run the demo after the installation to confirm that everything works as expected.


## reference

### dataset

[Rico](https://interactionmining.org/rico) / [Rico UI Screenshots and View Hierarchies dataset](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz)

Screen2Words: [paper](https://arxiv.org/abs/2108.03353) / [code](https://github.com/google-research/google-research/tree/master/screen2words) / [dataset](https://github.com/google-research-datasets/screen2words)
