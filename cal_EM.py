
import sys
import pandas as pd





def compute_exact_match(predictions, references):
    exact_matches = []
    for pred, refs in zip(predictions, references):
        if pred in refs:
            print(pred)
            print(refs)
            exact_matches.append(1)
        else:
            exact_matches.append(0)
    return sum(exact_matches) / len(exact_matches) * 100  


df = pd.read_json('./dev_adaptive.json', lines=True)
given_answer = []
for i in range(len(df)):
    answer = df.iloc[i]['answer']
    given_answer.append(answer)

df1 = pd.read_json(
    './the_doc_with_best_sentence.json',
    lines=True)
generated_answer = []

for i in range(len(df1)):
    answer = df1.iloc[i]['answer']
    generated_answer.append(answer.strip())


if __name__ == '__main__':
    results = compute_exact_match(predictions=generated_answer, references=given_answer)

    print(f"Exact Match Score: {results:.2f}%")