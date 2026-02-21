# Text-to-SQL: Comparative Analysis of Zero-Shot Prompting vs. Fine-Tuned Language Models

A research project comparing different approaches to natural language to SQL query generation, evaluating how zero-shot prompting stacks up against parameter-efficient fine-tuning across different model architectures.

## Objective

The goal of this project is to explore and compare how well language models can translate natural language questions into SQL queries given a database schema. Specifically, we wanted to answer:

- Can a pre-trained model generate meaningful SQL without any task-specific training (zero-shot)?
- Does fine-tuning on a Text-to-SQL dataset improve query generation quality?
- How does model architecture (encoder-decoder vs. decoder-only) affect performance on this task?

## What We Did

We implemented and evaluated three approaches using the [Gretel AI Synthetic Text-to-SQL](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) dataset (100,000 examples across diverse domains):

1. **Zero-Shot Flan-T5** — Tested `google/flan-t5-base` with a structured schema + question prompt, no training involved. This serves as the baseline.

2. **Fine-Tuned Flan-T5 + LoRA** — Fine-tuned `google/flan-t5-base` on the Text-to-SQL task using Low-Rank Adaptation (LoRA), a parameter-efficient technique that trains only ~0.35% of model parameters.

3. **Fine-Tuned Phi-2 + LoRA** — Fine-tuned `microsoft/phi-2` (2.7B parameter decoder-only model) using LoRA to see how a larger, architecturally different model compares on the same task.

Models were evaluated on BLEU score, Exact Match, and SQL Validity. Full findings and analysis are documented in [`text_to_sql_report.md`](./text_to_sql_report.md).

## Project Structure

```
text-to-sql/
├── zero_shot_prompting.py      # Zero-shot evaluation with Flan-T5
├── fine_tune_flan_t5.py        # Fine-tuning Flan-T5 with LoRA
├── fine_tuning_phi2.py         # Fine-tuning Phi-2 with LoRA
├── requirements.txt            # Dependencies
├── text_to_sql_report.md       # Full comparison report
└── .gitignore
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run in order:

```bash
# 1. Zero-shot baseline (no training, fastest)
python zero_shot_prompting.py

# 2. Fine-tune Flan-T5
python fine_tune_flan_t5.py

# 3. Fine-tune Phi-2 (recommended to run on GPU/Kaggle)
python fine_tuning_phi2.py
```

> **Note:** Phi-2 is a 2.7B parameter model. It is recommended to run `fine_tuning_phi2.py` on a GPU (e.g., Kaggle or Google Colab) rather than a local CPU.

## Fine-Tuning Details

Both fine-tuned models use **LoRA** (Low-Rank Adaptation) with the following config:
- Rank: 8
- Alpha: 32
- Dropout: 0.1
- Trainable parameters: ~0.33% of total

Experiment tracking is handled via [Weights & Biases](https://wandb.ai).

## Requirements

- Python 3.10+
- PyTorch
- Hugging Face Transformers, Datasets, PEFT
- See `requirements.txt` for full list

## Author

**Tushar Vimalbhai Patel**  
Northeastern University
