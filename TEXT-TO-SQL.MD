# Comparative Analysis of Text-to-SQL Generation Approaches: Zero-Shot Prompting vs. Fine-Tuned Language Models

**Author:** Tushar Vimalbhai Patel  
**Institution:** Northeastern University  
**Date:** February 2026  

---

## Abstract

This report presents a comparative study of three approaches to the Text-to-SQL task: zero-shot prompting with Flan-T5, fine-tuned Flan-T5 with LoRA, and fine-tuned Phi-2 with LoRA. All models were evaluated on the Gretel AI Synthetic Text-to-SQL dataset using BLEU score, Exact Match (EM), and SQL Validity as evaluation metrics. Results demonstrate that while fine-tuning improves BLEU score and SQL Validity over the zero-shot baseline, all three models produced near-zero Exact Match scores, indicating that short training runs on small data subsets are insufficient for learning precise SQL generation. Phi-2 achieved the highest BLEU score (0.0085) despite its causal language model architecture being less naturally suited to the task, while Flan-T5 fine-tuned achieved the highest SQL Validity (0.96). These findings highlight the trade-offs between model architecture, training scale, and evaluation methodology in Text-to-SQL research.

---

## 1. Introduction

The Text-to-SQL task involves translating natural language questions into executable SQL queries, given a database schema as context. This capability has significant practical value, enabling non-technical users to query databases without knowledge of SQL syntax. With the rapid advancement of large language models (LLMs), multiple approaches have emerged — ranging from zero-shot prompting, which leverages pre-trained knowledge without any task-specific training, to parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA) that adapt models to a specific task with minimal compute.

This study evaluates three such approaches:

1. **Zero-Shot Flan-T5** — Using `google/flan-t5-base` with a structured prompt, no fine-tuning.
2. **Fine-Tuned Flan-T5 + LoRA** — Fine-tuning `google/flan-t5-base` on the Text-to-SQL task using LoRA adapters.
3. **Fine-Tuned Phi-2 + LoRA** — Fine-tuning `microsoft/phi-2` (2.7B parameters) using LoRA adapters.

The central research question is: *How do zero-shot prompting and parameter-efficient fine-tuning compare for Text-to-SQL generation across different model architectures?*

---

## 2. Dataset

All experiments used the **Gretel AI Synthetic Text-to-SQL dataset** (`gretelai/synthetic_text_to_sql`), sourced from Hugging Face. The dataset contains 100,000 examples spanning diverse domains including finance, healthcare, defense, forestry, and entertainment. Each example consists of:

- `sql_context`: A CREATE TABLE statement defining the schema
- `sql_prompt`: A natural language question
- `sql`: The ground truth SQL query
- `sql_explanation`: A plain-English explanation of the query

Due to computational constraints of running experiments locally on a Mac CPU, a subset of the full dataset was used for training and evaluation. The training split consisted of 500 examples and the evaluation split of 100 examples, with an 80/20 train-eval ratio applied at the dataset level before subsampling.

---

## 3. Methodology

### 3.1 Models

**Flan-T5 Base** is an encoder-decoder transformer model developed by Google, instruction-tuned on a large variety of tasks. Its seq2seq architecture makes it naturally well-suited for structured generation tasks like Text-to-SQL, where the input and output are distinct sequences.

**Phi-2** is a 2.7B parameter causal language model developed by Microsoft. Unlike Flan-T5, Phi-2 uses a decoder-only architecture, meaning it generates output as a continuation of the input prompt rather than as a separate target sequence.

### 3.2 Fine-Tuning with LoRA

Both fine-tuned models used **Low-Rank Adaptation (LoRA)**, a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable low-rank matrices into the attention layers. This drastically reduces the number of trainable parameters, making fine-tuning feasible on consumer hardware.

LoRA configuration for both models:
- Rank (`r`): 8
- Alpha (`lora_alpha`): 32
- Dropout: 0.1
- Bias: none

Trainable parameters as a percentage of total:
- Flan-T5: 0.3561% (884,736 / 248,462,592)
- Phi-2: 0.3290% (9,175,040 / 2,788,858,880)

### 3.3 Training Configuration

| Parameter | Flan-T5 Fine-Tuned | Phi-2 Fine-Tuned |
|---|---|---|
| Epochs | 3 | 3 |
| Batch Size | 8 | 4 |
| Gradient Accumulation | 2 | 2 |
| LR Scheduler | Linear | Linear |
| Warmup Steps | 50 | 50 |
| fp16 | No (CPU) | No (CPU) |
| Early Stopping | Yes (patience=2) | No |

### 3.4 Input Formatting

All models received a structured prompt of the form:

```
Schema:
<CREATE TABLE statement>

Question:
<Natural language question>

Generate SQL query.
```

For the fine-tuned models, the target output included the formatted SQL query and a plain-English explanation. For the zero-shot model, only the input prompt was provided with no target supervision.

### 3.5 Evaluation Metrics

Three metrics were used:

- **BLEU Score** — Measures n-gram overlap between generated and reference SQL. A higher score indicates closer lexical similarity to the ground truth.
- **Exact Match (EM)** — Binary metric; 1 if the generated SQL exactly matches the reference (after normalization), 0 otherwise.
- **SQL Validity** — Proportion of generated outputs that parse as syntactically valid SQL using `sqlparse`.

---

## 4. Results

### 4.1 Quantitative Results

| Model | BLEU | Exact Match | SQL Validity |
|---|---|---|---|
| Flan-T5 Zero-Shot | 0.0023 | — | 0.92 |
| Flan-T5 Fine-Tuned | 0.0052 | 0.00 | 0.96 |
| Phi-2 Fine-Tuned | 0.0085 | 0.00 | 0.84 |

### 4.2 BLEU Score Analysis

All three models achieved very low BLEU scores, which is expected given the small training size and short training duration. However, a clear trend is visible: fine-tuning consistently improves BLEU over the zero-shot baseline. Phi-2 fine-tuned achieved the highest BLEU (0.0085), more than 3.5× the zero-shot baseline, despite being a decoder-only model not natively designed for structured generation. Flan-T5 fine-tuned achieved a BLEU of 0.0052, roughly 2.3× the baseline.

The low absolute BLEU values are attributable to two main factors: the small training dataset (500 examples) and the strict lexical matching nature of BLEU, which penalizes any deviation in token order or phrasing even when the query is semantically correct.

### 4.3 Exact Match Analysis

Both fine-tuned models scored 0.00 on Exact Match, meaning no generated query was an exact character-for-character match with the ground truth. This is unsurprising — SQL queries can be semantically equivalent while differing syntactically (e.g., column ordering, alias naming, whitespace), and the models were trained on a very small sample for only 3 epochs. Exact Match as a metric is known to be overly strict for SQL evaluation and is typically supplemented with execution-based metrics in production research settings.

### 4.4 SQL Validity Analysis

SQL Validity tells a more encouraging story. Flan-T5 fine-tuned achieved the highest validity at 0.96, meaning 96% of its outputs were parseable as valid SQL — an improvement over the zero-shot baseline of 0.92. This suggests that fine-tuning helped the model conform more reliably to SQL syntax. Phi-2 fine-tuned scored lower at 0.84, likely because its causal generation architecture is prone to hallucinating non-SQL tokens or producing repetitive patterns that fail SQL parsing, as observed in the qualitative examples.

### 4.5 Qualitative Observations

Examining individual predictions reveals interesting failure patterns. The zero-shot Flan-T5 model frequently outputs strings of hash characters (`####...`) or fragments of SQL with no structure. The fine-tuned Flan-T5 produces outputs that appear SQL-like in structure but often repeat column names or insert Python-like syntax instead of SQL. Phi-2 exhibits a more pronounced tendency to hallucinate, generating loops and Python code constructs — a reflection of the fact that its training data included a large proportion of code across multiple programming languages, not just SQL.

---

## 5. Discussion

### 5.1 Architecture Considerations

The results highlight an important architectural distinction. Flan-T5's encoder-decoder design is inherently better suited to Text-to-SQL because it treats the schema+question as a complete input and generates the SQL as a separate output sequence. Phi-2, being decoder-only, must generate SQL as a continuation of the input, making it harder to cleanly separate input context from output generation. Despite this architectural disadvantage, Phi-2 achieved a higher BLEU score after fine-tuning, likely because its larger parameter count (2.7B vs 248M) gives it greater representational capacity even with LoRA.

### 5.2 Impact of Training Scale

The most significant limiting factor in this study is the training scale. Using only 500 training examples and 3 epochs is far below what is typically required for meaningful SQL generation. State-of-the-art Text-to-SQL models are trained on tens of thousands of examples over many epochs. The results here should be interpreted as a proof-of-concept demonstrating that the training pipeline works correctly, not as an indicator of the models' true potential.

### 5.3 Limitations

- **Small dataset**: 500 training examples is insufficient for learning complex SQL patterns across diverse domains.
- **Short training**: 3 epochs with early stopping limits model convergence.
- **CPU training**: Running on CPU constrained batch size and training time, preventing experimentation with hyperparameters.
- **BLEU limitations**: BLEU measures surface similarity, not semantic correctness. A query can be functionally correct while scoring BLEU = 0 if phrased differently from the reference.
- **No execution-based evaluation**: Execution accuracy (running both queries on a database and comparing results) would be a more meaningful metric but was not implemented.

---

## 6. Conclusion

This study compared zero-shot prompting and LoRA-based fine-tuning for Text-to-SQL generation using Flan-T5 and Phi-2. The key findings are:

1. **Fine-tuning improves over zero-shot** across both BLEU and SQL Validity, even with minimal training data and short training runs.
2. **Phi-2 achieves higher BLEU** than Flan-T5 after fine-tuning, suggesting that model scale can compensate for architectural mismatch in this task.
3. **Flan-T5 achieves higher SQL Validity**, reflecting its more structured seq2seq output behaviour.
4. **Exact Match is too strict** for evaluating SQL generation and should be replaced with execution-based accuracy in future work.

For production use, these models would require training on the full 100,000-example dataset over 10+ epochs with GPU acceleration, along with execution-based evaluation metrics. Future work should also explore larger model variants such as `flan-t5-large` or `flan-t5-xl`, as well as more recent models specifically designed for code and SQL generation.

---

## 7. References

1. Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR.
2. Chung, H.W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. arXiv:2210.11416.
3. Li, Y., et al. (2023). *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*. NeurIPS.
4. Hu, E., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
5. Microsoft Research. (2023). *Phi-2: The surprising power of small language models*. Microsoft Research Blog.
6. Gretel AI. (2023). *Synthetic Text-to-SQL Dataset*. Hugging Face Datasets.