import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments,EvalPrediction
from sklearn.metrics import accuracy_score
import numpy as np



class CustomSeq2SeqTrainer(Trainer):
    def log(self, logs: dict):
        if self.is_local_process_zero():
            self.store_flos()
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.state.log_history.append({**logs, "step": self.state.global_step})
            with open('train_log.txt', 'a') as f:
                f.write(str(logs) + '\n')


class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        documents = " ".join(item['documents'])
        input_text = f"query: {query} documents: {documents}"
        label = item['label']

        encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Convert label to tensor
        label_encoding = torch.tensor([label], dtype=torch.long)  # Ensure label is an integer tensor

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_encoding
        }

def load_data(file_path, tokenizer, max_length=2048):
    with open(file_path, 'r') as f:
        data = json.load(f)
    dataset = TextClassificationDataset(data, tokenizer, max_length)
    return dataset

def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    predictions = np.argmax(preds, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_preds = [clean_output(dp) for dp in decoded_preds]
    print(decoded_preds)
    labels = p.label_ids

    print("Predictions:", predictions)
    print("Labels:", labels)
    assert len(predictions) == len(labels), "Length of predictions and labels must match"


    accuracy = accuracy_score(labels, predictions)

    return {"accuracy": accuracy}



def clean_output(output: str) -> str:
    return output.strip()

file_path = './final_data.json'
model_path = './Models/flan-t5-large'


tokenizer = T5Tokenizer.from_pretrained(model_path)
dataset = load_data(file_path, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


model = T5ForConditionalGeneration.from_pretrained(model_path)


training_args = TrainingArguments(
    output_dir='./t5-checkpoint',
    num_train_epochs=16,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=7000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


trainer.train()


results = trainer.evaluate()
print("Validation Results:", results)


test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Results:", test_results)