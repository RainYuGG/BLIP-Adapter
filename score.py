import evaluate
import numpy as np

def transpose(arr):
    return [list(i) for i in zip(*arr)]

def calculate_bleu(predictions, references):
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    print("evaluate_bleu",results)
    return results['bleu']