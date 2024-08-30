import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import transformers
from tqdm import tqdm
from embedding_sentences import sorted_sentences
import re
import argparse


def get_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--input', type='str')
    parser.add_argument('--output', type='str')
    args = parser.parse_args()
    return args


def load_llama():
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
   
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    model = base_model.eval()
    return model, tokenizer


def load_qa():
    question = []
    answer = []
    documents = []
    with open(args.input, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                question.append(data['question'])
                answer.append(data['answer'])
                documents.append(data['documents'])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
    return question, answer, documents


def get_index(sentence, documents):
    for i, doc in enumerate(documents):
        if sentence in doc:
            return i + 1


def split_into_sentences(paragraph):
    sentences = re.split(r'(?<=[.!?]) +', paragraph)
    return sentences


if __name__ == '__main__':
    args=get_args()
    question, answer, documents = load_qa()
    model, tokenizer = load_llama()
    data = []
    for i in tqdm(range(len(question))):
        dict = {"question": question[i], "answer": answer[i], "documents": documents[i], 5: None, 4: None, 3: None,
                2: None, 1: None,
                0: None, 'final': None}
        sentences = []
        for doc in documents[i]:
            sentences.extend(split_into_sentences(doc))
        sorted_sentences_with_scores, sorted_final_sentences = sorted_sentences(question[i], sentences)
        number = get_index(sorted_final_sentences[0], dict['documents'])
        dict['best_sentence'] = sorted_final_sentences[0]
        dict['best_sentence_index'] = number
        for j in range(1, 6):
            # final_docs = '\n'.join(dict['documents'][0:j])
            final_docs = ""
            for s, item in enumerate(dict['documents'][0:j], start=1):
                final_docs += f"document {s}:{item}\n"
            system_prompt = f"You are a helpful, respectful and honest assistant, and please use documents provided to answer the query.\nProvided documents:\n{final_docs.strip()}"
            input = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{question[i]} [/INST]"

            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            sequences = pipeline(
                input,
                do_sample=True,
                top_k=5,
                top_p=0.75,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=9000,
                return_full_text=False,
                temperature=0.01
            )
            ans = sequences[0]['generated_text']
            # print(j, ans)
            dict[j] = ans.strip()
            if dict['final'] is None and ans.strip() in answer[i]:
                dict['final'] = j

        system_prompt = f"""You are a helpful, respectful and honest assistant, and please use your knowledge to answer the query.\n"""
        my_input = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{question[i]} [/INST]"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        sequences = pipeline(
            my_input,
            do_sample=True,
            top_k=5,
            top_p=0.75,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=6000,
            return_full_text=False,
            temperature=0.01
        )
        ans = sequences[0]['generated_text']
        dict[0] = ans.strip()
        if ans.strip() in answer[i]:
            dict['final'] = 0
        data.append(dict)
        with open(args.output, 'a+',
                  encoding='utf-8') as f:
            json.dump(dict, f)
            f.write('\n')
