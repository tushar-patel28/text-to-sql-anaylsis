# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import torch
import sqlparse
import wandb
import nltk
from datasets import Dataset, load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from sqlparse import tokens as T

nltk.download("punkt")

# ── Load Dataset ───────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset_hf = load_dataset("gretelai/synthetic_text_to_sql")
df = dataset_hf["train"].to_pandas()
print(f"Total rows: {len(df)}")

hf_dataset = Dataset.from_pandas(df).shuffle(seed=42)

train_size = int(0.8 * len(hf_dataset))
eval_dataset = hf_dataset.select(range(train_size, train_size + 100))
print(f"Eval: {len(eval_dataset)}")

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(example):
    input_text = f"Schema:\n{example['sql_context']}\n\nQuestion:\n{example['sql_prompt']}\n\nGenerate SQL query."
    target_text = f"Explanation: {example['sql_explanation']}\nSQL: {example['sql']}"
    return {"input": input_text.strip(), "target": target_text.strip()}

processed_eval = eval_dataset.map(preprocess)
processed_eval = processed_eval.select(range(50))  # Evaluate only 50 examples for speed

# ── Load Model ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\nZero-Shot Sample Predictions:")
smooth_fn = SmoothingFunction().method4
references = []
candidates = []
valid_sql_count = 0

for i in range(len(processed_eval)):
    raw_input = processed_eval[i]["input"]
    reference = processed_eval[i]["target"]

    input_ids = tokenizer(raw_input, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids=input_ids, max_length=64, num_beams=1)  # greedy decoding
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generated_sql = sqlparse.format(generated, reindent=True, keyword_case="upper")
    reference_sql = sqlparse.format(reference, reindent=True, keyword_case="upper")

    references.append([reference_sql.split()])
    candidates.append(generated_sql.split())

    try:
        parsed = sqlparse.parse(generated_sql)
        if len(parsed) > 0 and all(tok.ttype != T.Error for stmt in parsed for tok in stmt.flatten()):
            valid_sql_count += 1
    except:
        pass

    if i < 5:
        print(f"\nExample {i+1}:")
        print("Input:", raw_input)
        print("Generated:", generated_sql)
        print("Reference:", reference_sql)
        print("BLEU:", sentence_bleu([reference_sql.split()], generated_sql.split(), smoothing_function=smooth_fn))

# ── Scores ─────────────────────────────────────────────────────────────────────
overall_bleu = corpus_bleu(references, candidates, smoothing_function=smooth_fn)
sql_validity = valid_sql_count / len(processed_eval)

print(f"\nZero-Shot BLEU Score: {overall_bleu:.4f}")
print(f"Zero-Shot SQL Validity Score: {sql_validity:.4f}")

# ── WandB Logging ──────────────────────────────────────────────────────────────
wandb.login(key="7517c861d0545ea5a6a9fd8a3e082f9b1a3b3804")
wandb.init(project="text-to-sql-flant5", name="flan-t5-zero-shot")
wandb.log({"BLEU": overall_bleu, "SQL_Validity": sql_validity})
wandb.finish()