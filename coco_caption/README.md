Microsoft COCO Caption Evaluation
===================

Evaluation codes for MS COCO caption generation.

## Installation
### 1. Install the model
To install the required Stanford model, run the following command:
```bash
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
sudo chown [user_name] java
sudo chgrp [user_name] java
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
python Scorer.py
```
## Requirements ##
- java 1.8.0
- python 2 or 3
  - gensim

## Files ##
./
- cocoEvalCapDemo.py (demo script)

./annotation
- captions_val2014.json (MS COCO 2014 caption validation set)
- Visit MS COCO [download](http://mscoco.org/dataset/#download) page for more details.

./results
- captions_val2014_fakecap_results.json (an example of fake results for running demo)
- Visit MS COCO [format](http://mscoco.org/dataset/#format) page for more details.

./pycocoevalcap: The folder where all evaluation codes are stored.
- evals.py: The file includes COCOEavlCap class that can be used to evaluate results on COCO.
- tokenizer: Python wrapper of Stanford CoreNLP PTBTokenizer
- bleu: Bleu evalutation codes
- meteor: Meteor evaluation codes
- rouge: Rouge-L evaluation codes
- cider: CIDEr evaluation codes
- ciderR: CIDEr-R evaluation codes
- spice: SPICE evaluation codes
- wmd: Word Mover's Distance evaluation codes

## Setup ##

- You will first need to download the [Stanford CoreNLP 3.6.0](http://stanfordnlp.github.io/CoreNLP/index.html) code and models for use by SPICE. To do this, run:
    bash get_stanford_models.sh
- Note: SPICE will try to create a cache of parsed sentences in ./pycocoevalcap/spice/cache/. This dramatically speeds up repeated evaluations. The cache directory can be moved by setting 'CACHE_DIR' in ./pycocoevalcap/spice. In the same file, caching can be turned off by removing the '-cache' argument to 'spice_cmd'. 
- You will also need to download the Google News negative 300 word2vec model for use by WMD. To do this, run:
    bash get_google_word2vec_model.sh


### AllSPICE
AllSPICE is a metric measuring both diversity and accuracy of a generated caption set. This is proposed in [Analysis of diversity-accuracy tradeoff in image captioning](https://arxiv.org/abs/2002.11848).

See [cocoEvalAllSPICEDemo.ipynb](cocoEvalAllSPICEDemo.ipynb) to learn how to use it.

You can also check out [ruotianluo/self-critical.pytorch/eval_multi.py](https://github.com/ruotianluo/self-critical.pytorch/blob/master/eval_multi.py#L36) to see how it is used in practice and [ruotianluo/SPICE](https://github.com/ruotianluo/SPICE/commit/046d8cf73db93653c0e22fdd0ed65d2106642244) to see what change was made to the original SPICE code to realize AllSPICE.

## References ##

- [Microsoft COCO Captions: Data Collection and Evaluation Server](http://arxiv.org/abs/1504.00325)
- PTBTokenizer: We use the [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml) which is included in [Stanford CoreNLP 3.4.1](http://nlp.stanford.edu/software/corenlp.shtml).
- BLEU: [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf)
- Meteor: [Project page](http://www.cs.cmu.edu/~alavie/METEOR/) with related publications. We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor). Changes have been made to the source code to properly aggreate the statistics for the entire corpus.
- Rouge-L: [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf)
- CIDEr: [CIDEr: Consensus-based Image Description Evaluation](http://arxiv.org/pdf/1411.5726.pdf)
- CIDEr-R: [CIDEr-R: Robust Consensus-based Image Description Evaluation](https://aclanthology.org/2021.wnut-1.39.pdf)
- SPICE: [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08822)
- WMD: [From word embeddings to document distances](http://proceedings.mlr.press/v37/kusnerb15.html) (original metric publication) and [Re-evaluating Automatic Metrics for Image Captioning](http://aclweb.org/anthology/E17-1019) (publication with metric adapted for caption generation)

Also,

- Stop words distributed by the NLTK Stopwords Corpus [nltk.corpus.stopwords.words('english')], which originate from [https://anoncvs.postgresql.org/cvsweb.cgi/pgsql/src/backend/snowball/stopwords/] and later augmented at [https://github.com/nltk/nltk_data/issues/22], were extracted and put in a text file in pycocoevalcap/wmd/data to avoid requiring users to install NLTK.
- Special thanks to David Semedo [https://github.com/davidfsemedo/coco-caption] for writing a Python 3 compatible version of coco-caption first and which was used as a reference to help make this fork.

## Developers ##
- Xinlei Chen (CMU)
- Hao Fang (University of Washington)
- Tsung-Yi Lin (Cornell)
- Ramakrishna Vedantam (Virgina Tech)

## Acknowledgement ##
- David Chiang (University of Norte Dame)
- Michael Denkowski (CMU)
- Alexander Rush (Harvard University)
- Mert Kilickaya (Hacettepe University)