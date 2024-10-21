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
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_function(examples):
    inputs = examples['Prompt']
    targets = examples['Response']
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=512)
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def token_level_accuracy_metric(preds, labels):
    total_tokens = 0
    matching_tokens = 0

    for pred, label in zip(preds, labels):
        pred_tokens = tokenizer.tokenize(pred)
        label_tokens = tokenizer.tokenize(label)

        total_tokens += len(label_tokens)
        matching_tokens += sum(1 for p, l in zip(pred_tokens, label_tokens) if p == l)

    token_accuracy = (matching_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0
    return token_accuracy

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=False, token='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')
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto"
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Load LoRA config and apply to the model
target_modules = ["q_proj", "v_proj"]
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_modules
)
model = get_peft_model(model, lora_config)

# Load dataset
df = pd.read_csv('../data/task_prompts1021.csv')
dataset = Dataset.from_pandas(df)
split_dataset = dataset.train_test_split(test_size=0.1)
split_dataset['train'] = split_dataset['train'].shuffle(seed=42)
split_dataset['test'] = split_dataset['test'].shuffle(seed=42)

# Tokenize dataset
processed_dataset = split_dataset.map(preprocess_function, batched=True)
tokenized_datasets = processed_dataset.remove_columns(["Prompt", "Response"])
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(72))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(8))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=1)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=1)

# Define evaluation metric
metric = evaluate.load("accuracy")

# Optimizer and training setup
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 50
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
progress_bar = tqdm(range(num_training_steps))

# Load sentence transformer model for STS
sts_model = SentenceTransformer('all-MiniLM-L6-v2')

epoch_total_sts = []
epoch_total_token_acc = []

model.train()
for epoch in tqdm(range(num_epochs)):
    ep_tot_sts = 0
    ep_tot_tok_acc = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

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

        # Compute exact match
        acc_result = token_level_accuracy_metric(decoded_preds, decoded_labels)
        print(f"Token Accuracy: {acc_result}%")

        # Compute Semantic Textual Similarity (STS) using BERT embeddings
        ref_embeddings = sts_model.encode(decoded_labels, convert_to_tensor=True)
        pred_embeddings = sts_model.encode(decoded_preds, convert_to_tensor=True)

        # Compute cosine similarity between predicted and reference answers
        cosine_sim = util.pytorch_cos_sim(ref_embeddings, pred_embeddings)
        print(f"Semantic Textual Similarity (STS) Scores: {cosine_sim}")

        # Print input, prediction, and similarity score
        decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        for input_text, pred_text, label_text, sim_score in zip(decoded_inputs, decoded_preds, decoded_labels, cosine_sim.diag()):
            print(f"Prompt: {input_text}")
            print(f"Prediction: {pred_text}")
            print(f"Reference: {label_text}")
            print(f"STS Similarity: {sim_score:.4f}")
            print("-" * 50)
        ep_tot_sts += sim_score
        ep_tot_tok_acc += acc_result
    
    # Average sts and token accuracy
    av_sts = (ep_tot_sts / len(batch)) 
    epoch_total_sts.append(av_sts)
    av_acc = (ep_tot_tok_acc / len(batch)) 
    epoch_total_token_acc.append(av_acc)

# Ensure that all items are either floats or moved to CPU and converted to numpy
epoch_total_token_acc = [metric.cpu().numpy() if isinstance(metric, torch.Tensor) else metric for metric in epoch_total_token_acc]
epoch_total_sts = [metric.cpu().numpy() if isinstance(metric, torch.Tensor) else metric for metric in epoch_total_sts]

# Create the DataFrame
eval_metrics_df = pd.DataFrame({
    'STS': epoch_total_sts, 
    'Token_Accuracy': epoch_total_token_acc
})

eval_metrics_df.to_csv('evaluation_results/long_form_response.csv')

print(f'\nFinished traingin: {num_epochs} epochs completed.\n------------DONE-----------')
