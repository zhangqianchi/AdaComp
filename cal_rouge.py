import json

import evaluate
import pandas as pd
import sys

rouge = evaluate.load('./metric/rouge')

if __name__ == '__main__':
    predictions = []
    references = []
    path = './RAG4WSDM-data/top5_doc.json'
    data = [json.loads(line.strip()) for line in open(path, 'r', encoding='utf-8')]
    for item in data:
        predictions.append(item['output_answer'])
        references.append(item['given_answer'])
    results = rouge.compute(predictions=predictions, references=references)
    print(results)
