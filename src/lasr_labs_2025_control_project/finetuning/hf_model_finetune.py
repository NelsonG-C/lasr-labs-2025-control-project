"""Fine-tuning script used to train gpt-oss-20b and gpt-oss-120b on binary self-recognition task. 
This was run on a single RTX 6000 Blackwell GPU, and only 4-bit quantization fit. """

import torch, unsloth 
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from unsloth import FastLanguageModel 


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-120b-unsloth-bnb-4bit", # use unsloth's 4-bit quantization because MXFP4 is still inference-only 
    max_seq_length=2048, 
    dtype=None, 
    load_in_4bit=True, 
    full_finetuning=False, 
)

# gpt-oss should already have a pad token, but fall back to eos just in case.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0,
    bias="none", 
)
model = get_peft_model(model, lora_config)

data = load_dataset(
    "json",
    data_files="/path/to/data_files",
)

MAX_LEN = 2048  


def preprocess(example):
    messages = example["messages"]

    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],                
        add_generation_prompt=True,   
        tokenize=True,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,                    
        tokenize=True,
    )

    # Truncate if needed.
    full_ids = full_ids[:MAX_LEN]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }

data = data.map(preprocess, remove_columns=data["train"].column_names)

split = data["train"].train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
test_data = split["test"]

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

model.config.use_cache = False 

training_args = TrainingArguments(
    output_dir="/path/to/output-dir",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True, 
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,          # match the bfloat16 weights; fp16=True can cause NaNs here
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("selfrec_adapter_2")
tokenizer.save_pretrained("selfrec_adapter_2")
