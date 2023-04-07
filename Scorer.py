# -*- coding=utf-8 -*-
# author: w61
# Test for several ways to compute the score of the generated words.
from coco_caption import pycocoevalcap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import evaluate
import numpy as np
# from pycocoevalcap.spice.spice import Spice
# from pycocoevalcap.wmd.wmd import WMD

class Scorers():
    def __init__(self,ref,gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
    

    def transform(self):
        # Initialize empty dictionaries to hold the transformed reference and prediction data
        ref_dict = {}
        pred_dict = {}

        for i, sublist in enumerate(self.gt):
            key = str(i+1)
            ref_dict[key] = sublist

        for i, sublist in enumerate(self.ref):
            key = str(i+1)
            pred_dict[key] = [sublist]
        
        # Overwrite the original gt and ref data with the transformed dictionaries
        self.gt = ref_dict
        self.ref = pred_dict


    def compute_scores(self):
        self.transform()
        total_scores = {}
        for scorer, method in self.scorers:
            # print('computing %s score...'%(scorer.method()))
            
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                # for sc, scs, m in zip(score, scores, method):
                #     print("%s: %0.3f"%(m, sc))
                total_scores["Bleu"] = score
            else:
                # print("%s: %0.3f"%(method, score))
                total_scores[method] = score
        
        # print('*****DONE*****')
        for key,value in total_scores.items():
            print('{}:{}'.format(key,value))

def transpose(arr):
    return [list(i) for i in zip(*arr)]

def calculate_score(predictions, references, metric = 'bleu'):
    score = evaluate.load(metric)
    results = score.compute(predictions=predictions, references=references)
    print(f"evaluate_{metric}: ",results)
    if(metric == 'rouge'):
        return results['rougeL']
    return results[metric]

if __name__ == '__main__':
    ref = ['go down the stairs and stop at the bottom .',
        'this is a cat.']
    gt = [
        ['Walk down the steps and stop at the bottom. ', 'Go down the stairs and wait at the bottom.','Once at the top of the stairway, walk down the spiral staircase all the way to the bottom floor. Once you have left the stairs you are in a foyer and that indicates you are at your destination.'],
        ['It is a cat.','There is a cat over there.','cat over there.']
    ]
    scorer = Scorers(ref,gt)
    scorer.compute_scores()
