from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import os

# Access the token from the environment variable
hf_token = os.getenv("HF_TOKEN")

def preprocess_function(prompt):
    # Tokenize the input prompt and return input_ids and attention_mask
    model_inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return model_inputs

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=False, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto"
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"DEVICE USED : {device}")

# Move model to the correct device (CPU/GPU)
model.to(device)

# Get user input
query = input("Prompt: ")
query = preprocess_function(query)

# Move inputs to the device
input_ids = query['input_ids'].to(device)
attention_mask = query['attention_mask'].to(device)

with torch.no_grad():
    # Generate response using the model
    outputs = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=100  
    )

# Decode predictions
decoded_preds = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Model's response: {decoded_preds}")
