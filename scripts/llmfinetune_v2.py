import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, get_scheduler
from peft import LoraConfig, get_peft_model
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import os
from tqdm.auto import tqdm

import numpy as np
import evaluate

# Access the token from the environment variable
hf_token = os.getenv("HF_TOKEN")

def preprocess_function(examples):
    # inputs = examples['input']
    # targets = examples['output']
    inputs = examples['Question']
    targets = examples['Answer']
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=512)
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(f'Logits : {logits}')
    print(f'Labels: {labels}')
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)

# Define a function to compute exact match
def exact_match_metric(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute exact match
    exact_matches = sum([1 if pred == label else 0 for pred, label in zip(decoded_preds, decoded_labels)])
    
    # Return EM score as a percentage
    return {"exact_match": exact_matches / len(decoded_preds) * 100}

    



tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=False, token=hf_token) 
# os.environ['CUDA_LAUNCH_BLOCKING'] = 1
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto"
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

target_modules = ["q_proj", "v_proj"]

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_modules
)

model = get_peft_model(model, lora_config)

with open(r"task_prompts.txt", encoding='utf-8') as file:
    lines = file.readlines()

data = []
question, answer = '', ''
collecting_answer = False

df = pd.read_csv('task_prompts.csv')
dataset = Dataset.from_pandas(df)

split_dataset = dataset.train_test_split(test_size=0.1)

split_dataset['train'] = split_dataset['train'].shuffle(seed=42)
split_dataset['test'] = split_dataset['test'].shuffle(seed=42)

# Tokenize Dataset
processed_dataset = split_dataset.map(preprocess_function, batched=True)
tokenized_datasets = processed_dataset.remove_columns(["Question"])
tokenized_datasets = tokenized_datasets.remove_columns(['Answer'])

print(tokenized_datasets)
# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(87))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=1)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=1)

metric = evaluate.load("accuracy")

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 50
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in tqdm(range(num_epochs)):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        # Decode the input prompt (Question)
        decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    
        # Print the input and corresponding prediction
        for input_text, pred_text in zip(decoded_inputs, decoded_preds):
            print(f"Prompt: {input_text}")
            print(f"Prediction: {pred_text}")
            print("-" * 50)  # Separator line for readability
            # Compute Exact Match (EM)
            em_result = exact_match_metric((logits.cpu(), batch['labels'].cpu()))
            print(f"Exact Match: {em_result['exact_match']}%")

        # # Ensure predictions and labels are in int32 format
        # predictions = predictions.cpu().numpy().astype("int32")
        # references = batch["labels"].cpu().numpy().astype("int32")
        # metric.add_batch(predictions=predictions, references=batch["labels"])

        # met = metric.compute()
        # print(f"Metric output: {met}")