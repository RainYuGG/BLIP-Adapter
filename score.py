import evaluate
import numpy as np

def transpose(arr):
    return [list(i) for i in zip(*arr)]

def calculate_score(predictions, references, metric = 'bleu'):
    score = evaluate.load(metric)
    results = score.compute(predictions=predictions, references=references)
    print(f"evaluate_{metric}: ",results)
    if(metric == 'rouge'):
        return results['rougeL']
    return results[metric]