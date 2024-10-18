import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, get_scheduler
from peft import LoraConfig, get_peft_model
import torch
from torch.optim import AdamW
import os

import numpy as np
import evaluate


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
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)



tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=False, token='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') #maybe you can try 2b first
# os.environ['CUDA_LAUNCH_BLOCKING'] = 1
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto"
)

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

# # Parse the file content
# for line in lines:
#     line = line.strip()

#     if line.startswith("Question:"):
#         if question and answer:  
#             data.append({'input': question, 'output': answer})
#         question = line.split(":", 1)[1].strip()  
#         answer = ''  
#         collecting_answer = True 
#     elif line.startswith("Answer:"):
#         collecting_answer = True 
#     else:
#         if collecting_answer:
#             answer += ' ' + line 

# if question and answer:
#     data.append({'input': question, 'output': answer})

# df = pd.DataFrame(data)
df = pd.read_csv('task_prompts.csv')
dataset = Dataset.from_pandas(df)

split_dataset = dataset.train_test_split(test_size=0.1)

split_dataset['train'] = split_dataset['train'].shuffle(seed=42)
split_dataset['test'] = split_dataset['test'].shuffle(seed=42)


processed_dataset = split_dataset.map(preprocess_function, batched=True)

metric = evaluate.load("accuracy")

training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Adjust batch size based on your system performance
    num_train_epochs=50,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    fp16=False,
    bf16=False,
    logging_dir='./logs',
    logging_steps=20,
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = training_args.num_train_epochs * len(processed_dataset['train'])
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['test'],
    tokenizer=tokenizer,
    optimizers=(optimizer, lr_scheduler),
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained('./finetuned_model')
tokenizer.save_pretrained('./finetuned_model')
model.to(torch.bfloat16)

# https://huggingface.co/docs/transformers/training
