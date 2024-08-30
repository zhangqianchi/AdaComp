import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import transformers
from tqdm import tqdm
from embedding_sentences import sorted_sentences
import random
import math




# 加载llama2-7b-hf的模型
def load_llama():
    tokenizer = AutoTokenizer.from_pretrained('./compressor_final',
                                              trust_remote_code=True)
    # 填充token赋值
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        "./compressor_final",
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    model = base_model.eval()
    return model, tokenizer


def load_qa_documents():
    question = []
    answer = []
    documents = []
    best_sentence = []
    best_sentence_index = []
    final = []
    with open('./output/nq_test_ordered_adaptive.json', 'r',
              encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            question.append(item['question'])
            answer.append(item['answer'])
            documents.append(item['documents'])
            best_sentence.append(item['best_sentence'])
            best_sentence_index.append(item['best_sentence_index'])
            final.append(item['final'])

    return question, answer, documents, best_sentence, best_sentence_index, final


def compressor():
    number = 0
    total_number = 0
    predict_list = []
    question, answer, documents, best_sentence, best_sentence_index, final = load_qa_documents()
    model, tokenizer = load_llama()
    for i in tqdm(range(len(question))):
        final_docs = ""
        for j, item in enumerate(documents[i], start=1):
            final_docs += f"document {j}:{item}\n"

        # system_prompt = f"Please output the top documents needed based on the complexity of the question and the document where the best sentence is located. If it is a one-hop question, one document is needed; if it is a two-hop question, two documents are needed, and so on. If the document where the best sentence is located has a higher index than the complexity of the question, then output the document where the best sentence is located.\nProvided documents:\n{final_docs}Best sentences:{best_sentence[i]}\n"
        system_prompt = f"""Please output the top documents needed based on the complexity of the question and the quality of the ducoments. If the question is complex or the quality of the documents is poor, the answer is high. If the question is easy or the quality of the documents is good, the answer is low.\nProvided documents:\n{final_docs}""",

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
        ans = str(int(math.ceil(float(ans))))
        print(ans)
        predict_list.append(ans)
        with open('./predic_list.txt','a+',encoding='utf-8') as f:
            f.write(ans+'\n')
    return predict_list

if __name__ == '__main__':
    total_number = compressor()

