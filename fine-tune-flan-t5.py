# ── Imports ───────────────────────────────────────────────────────────────────
import re
import pandas as pd
import torch
import sqlparse
import wandb
import nltk
from datasets import Dataset, load_dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
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
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([?.!,])", r"\1", text)
    return text

def preprocess(example):
    formatted_sql = sqlparse.format(
        example["sql"],
        keyword_case="upper",
        identifier_case="lower",
        strip_comments=True,
        reindent=True,
        indent_columns=True,
        use_space_around_operators=True
    ).strip()

    explanation = normalize_text(example["sql_explanation"])
    input_text = f"Schema:\n{example['sql_context'].strip()}\n\nQuestion:\n{example['sql_prompt'].strip()}\n\nGenerate SQL query."
    target_text = f"SQL Query:\n{formatted_sql}\n\nExplanation:\n{explanation}"
    return {"input": input_text.strip(), "target": target_text.strip()}

processed_train = train_dataset.map(preprocess)
processed_eval = eval_dataset.map(preprocess)

# ── Tokenizer ──────────────────────────────────────────────────────────────────
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)

def tokenize(example):
    model_inputs = tokenizer(
        example["input"], padding="max_length", truncation=True, max_length=96
    )
    labels = tokenizer(
        example["target"], padding="max_length", truncation=True, max_length=64
    )
    labels["input_ids"] = [
        -100 if token_id == tokenizer.pad_token_id else token_id
        for token_id in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = processed_train.map(tokenize, batched=True)
tokenized_eval = processed_eval.map(tokenize, batched=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
base_model = T5ForConditionalGeneration.from_pretrained(model_name)
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1,
    bias="none", task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ── Training ───────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    lr_scheduler_type="linear",
    logging_strategy="steps",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# ── Save Model ─────────────────────────────────────────────────────────────────
save_path = "./lora-flan-t5"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to {save_path}")

# ── WandB Init ─────────────────────────────────────────────────────────────────
wandb.init(
    project="text-to-sql-flant5",
    name="flan-t5-finetuned",
    config={
        "model": "flan-t5-base + LoRA (fine-tuned)",
        "dataset": "Gretel synthetic Text-to-SQL",
        "eval_size": len(processed_eval),
        "num_beams": 5,
        "max_length": 64,
        "padding": "max_length"
    }
)

# ── Helper Functions ───────────────────────────────────────────────────────────
def extract_sql_only(text):
    match = re.search(r"SQL Query:\n(.+?)(\n\nExplanation:|$)", text, re.DOTALL)
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
print("\nSample Predictions:")
smooth_fn = SmoothingFunction().method4
references = []
candidates = []
em_matches = 0
valid_sql_count = 0
total = len(processed_eval)

for i in range(total):
    raw_input = processed_eval[i]["input"]
    input_ids = tokenizer(raw_input, return_tensors="pt").input_ids.to(model.device)

    output_ids = model.generate(
        input_ids=input_ids, max_length=128, num_beams=5, early_stopping=True
    )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reference = processed_eval[i]["target"]

    generated_sql = sqlparse.format(extract_sql_only(generated), reindent=True, keyword_case="upper")
    reference_sql = sqlparse.format(extract_sql_only(reference), reindent=True, keyword_case="upper")

    references.append([reference_sql.split()])
    candidates.append(generated_sql.split())

    if generated_sql.strip().lower() == reference_sql.strip().lower():
        em_matches += 1
    if is_valid_sql(generated_sql):
        valid_sql_count += 1

    if i < 5:
        print(f"\nExample {i+1}")
        print("Input:", raw_input)
        print("Reference Output:\n", reference.strip())
        print("Fine-tuned Output:\n", generated.strip())
        print("BLEU:", sentence_bleu([reference_sql.split()], generated_sql.split(), smoothing_function=smooth_fn))

# ── Scores ─────────────────────────────────────────────────────────────────────
overall_bleu = corpus_bleu(references, candidates, smoothing_function=smooth_fn)
em_score = em_matches / total
sql_validity = valid_sql_count / total

print(f"\nOverall BLEU: {overall_bleu:.4f}")
print(f"Exact Match (EM): {em_score:.4f}")
print(f"SQL Validity: {sql_validity:.4f}")

wandb.log({"BLEU": overall_bleu, "EM": em_score, "SQL_Validity": sql_validity})
wandb.finish()