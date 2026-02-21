# ── Imports ───────────────────────────────────────────────────────────────────
import re
import torch
import pandas as pd
import sqlparse
import wandb
import nltk
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from sqlparse import tokens as T

nltk.download("punkt")

# ── WandB ──────────────────────────────────────────────────────────────────────
wandb.login(key="7517c861d0545ea5a6a9fd8a3e082f9b1a3b3804")

# ── Load Dataset ───────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset_hf = load_dataset("gretelai/synthetic_text_to_sql")
df = dataset_hf["train"].to_pandas()
print(f"Total rows: {len(df)}")

hf_dataset = Dataset.from_pandas(df).shuffle(seed=42)

train_dataset = hf_dataset.select(range(500))
eval_dataset = hf_dataset.select(range(500, 600))
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# ── Preprocessing ──────────────────────────────────────────────────────────────
def normalize_text(text):
    return sqlparse.format(text.strip(), keyword_case="upper", strip_comments=True)

def preprocess(example):
    input_text = f"Schema:\n{example['sql_context']}\n\nQuestion:\n{example['sql_prompt']}\n\nGenerate SQL query."
    target_text = f"SQL: {normalize_text(example['sql'])} || Explanation: {example['sql_explanation']}"
    return {"input": input_text.strip(), "target": target_text.strip()}

processed_train = train_dataset.map(preprocess)
processed_eval = eval_dataset.map(preprocess)

# ── Tokenizer ──────────────────────────────────────────────────────────────────
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    inputs = tokenizer(example["input"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(example["target"], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = [
        -100 if token_id == tokenizer.pad_token_id else token_id
        for token_id in targets["input_ids"]
    ]
    return inputs

tokenized_train = processed_train.map(tokenize, batched=True)
tokenized_eval = processed_eval.map(tokenize, batched=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
device = "cpu"  # MPS has known issues with Phi-2 + LoRA on Mac
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,  # float32 required for CPU
)
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1,
    bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ── Training ───────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    gradient_accumulation_steps=2,
    lr_scheduler_type="linear",
    warmup_steps=50,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="none",
    fp16=False  # Must be False on CPU/Mac
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    processing_class=tokenizer,
    data_collator=data_collator
)

trainer.train()

# ── Save Model ─────────────────────────────────────────────────────────────────
save_path = "./phi2-lora"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to {save_path}")

# ── Helper Functions ───────────────────────────────────────────────────────────
def extract_sql_only(text):
    match = re.search(r"SQL:\s*(.*?)\s*\|\|", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def is_valid_sql(query):
    try:
        parsed = sqlparse.parse(query)
        return len(parsed) > 0 and all(
            token.ttype != T.Error for stmt in parsed for token in stmt.flatten()
        )
    except:
        return False

# ── Evaluation ─────────────────────────────────────────────────────────────────
smooth_fn = SmoothingFunction().method4
references = []
candidates = []
em_matches = 0
valid_sql_count = 0

subset = processed_eval.select(range(50))

with torch.no_grad():
    for ex in subset:
        encodings = tokenizer(
            ex["input"], return_tensors="pt",
            padding="max_length", truncation=True, max_length=128
        )
        input_ids = encodings.input_ids.to(model.device)
        attention_mask = encodings.attention_mask.to(model.device)

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id
        )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        pred_sql = extract_sql_only(generated)
        ref_sql = extract_sql_only(ex["target"])

        pred_sql_clean = sqlparse.format(pred_sql, reindent=True, keyword_case="upper")
        ref_sql_clean = sqlparse.format(ref_sql, reindent=True, keyword_case="upper")

        references.append([ref_sql_clean.split()])
        candidates.append(pred_sql_clean.split())

        if pred_sql_clean.strip().lower() == ref_sql_clean.strip().lower():
            em_matches += 1
        if is_valid_sql(pred_sql_clean):
            valid_sql_count += 1

# ── Scores ─────────────────────────────────────────────────────────────────────
bleu = corpus_bleu(references, candidates, smoothing_function=smooth_fn)
em_score = em_matches / len(subset)
sql_validity = valid_sql_count / len(subset)

print(f"BLEU: {bleu:.4f}, EM: {em_score:.4f}, SQL Validity: {sql_validity:.4f}")