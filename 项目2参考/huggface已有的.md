Below are two distinct “quick‑start” solutions for the Recipe Recommendation module that you can drop into Colab and have working **in seconds** (Scheme A) or with a **small fine‑tuning run** (Scheme B). Both approaches meet the assignment requirements of taking an ingredient list and returning structured recipes (name, quantities, instructions).

---

## Summary

- **Scheme A: Zero‑Shot Pipeline**  
  Leverages an **already fine‑tuned** recipe generator on Hugging Face Hub so you **never train** yourself—just one `pipeline()` call and you get multiple recipes in < 1 s per prompt.  
- **Scheme B: Lightweight Fine‑Tuning**  
  Uses the Food_Recipe dataset but fine‑tunes only a **T5‑small** model for **1 epoch** (< 5 min on GPU) to tailor it to your data. Includes full Colab‑ready code, progress bars, and example generations.

---

## Scheme A: Zero‑Shot with Pre‑Trained Recipe Generator

### 1. Install & Import

```bash
!pip install --quiet transformers datasets
```
```python
from transformers import pipeline
```
> The Hugging Face [Pipeline API] offers plug‑and‑play inference for text2text tasks citeturn0search0turn0search1.

### 2. Load a Public Recipe Model

```python
generator = pipeline(
    "text2text-generation",
    model="flax-community/t5-recipe-generation"
)
```
> The `flax-community/t5-recipe-generation` (“Chef Transformer”) was trained on 2.2 M recipes and is fully public citeturn1search0turn1search4.

### 3. Generate Multiple Recipes

```python
ingredients = "tomato, basil, garlic"

results = generator(
    f"Suggest recipes given ingredients: {ingredients}.",
    max_length=128,
    num_beams=4,
    num_return_sequences=3
)

for i, out in enumerate(results, 1):
    print(f"--- Recipe {i} ---\n{out['generated_text']}\n")
```
- `num_return_sequences`: returns `k` distinct beams citeturn0search7  
- `num_beams`: controls diversity vs. quality citeturn1search5  

### 4. Why It’s Fast & Complete

1. **Zero‑Training**: no fine‑tuning steps required.  
2. **Lightweight Model**: T5‑small footprint (~80 M params) ensures < 1 s per call citeturn1search1.  
3. **Multi‑Recipe Output**: you get multiple candidate recipes in one shot.  

---

## Scheme B: Quick Fine‑Tune on Food_Recipe Dataset

### 1. Install & Setup

```bash
!pip install --quiet transformers datasets accelerate evaluate
```
```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
    Seq2SeqTrainer, set_seed
)
set_seed(42)  # ensure reproducibility citeturn2search6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 2. Load & Inspect Data

```python
ds = load_dataset("BhavaishKumar112/Food_Recipe")
print("Columns:", ds["train"].column_names)
# ['name','description','cuisine','course','diet',
#  'ingredients_name','ingredients_quantity',…,'instructions','image_url']
```

### 3. Split & Preprocess

```python
# field names
KT, KIN, KIQ, KI = "name", "ingredients_name", "ingredients_quantity", "instructions"

split = ds["train"].train_test_split(0.1, seed=42)
train_ds = split["train"].filter(lambda x: x[KIN] and x[KIQ] and x[KI])
eval_ds  = split["test"].filter(lambda x: x[KIN] and x[KIQ] and x[KI])

def preprocess(x):
    names, qtys = x[KIN], x[KIQ]
    instr = x[KI].replace("\n"," ")
    # prompt
    inp = f"Suggest a recipe given ingredients: {names}."
    # pair names & quantities
    n_list, q_list = names.split(","), qtys.split(",")
    pairs = zip(n_list,q_list) if len(n_list)==len(q_list) else [(n,"") for n in n_list]
    info = "; ".join([f"{n.strip()}: {q.strip()}" for n,q in pairs])
    tgt = (f"Recipe name: {x[KT]}\n"
           f"Ingredients & quantities: {info}\n"
           f"Instructions: {instr}")
    return {"input_text": inp, "target_text": tgt}

train_ds = train_ds.map(preprocess, remove_columns=ds["train"].column_names)
eval_ds  = eval_ds.map(preprocess,  remove_columns=ds["train"].column_names)
```
> Follows T5 fine‑tuning best practices (prefix “Suggest a recipe…” etc.) citeturn2search0turn2search7.

### 4. Tokenize & Collate

```python
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    inp = tokenizer(batch["input_text"],  max_length=128, truncation=True, padding="max_length")
    tgt = tokenizer(batch["target_text"], max_length=256, truncation=True, padding="max_length")
    inp["labels"] = tgt["input_ids"]
    return inp

train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["input_text","target_text"])
eval_tok  = eval_ds.map(tokenize_fn,  batched=True, remove_columns=["input_text","target_text"])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_name)
```
> Uses full‑padding for stable batches citeturn2search7.

### 5. Configure & Launch Training

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

training_args = Seq2SeqTrainingArguments(
    output_dir="recipe_model",
    eval_strategy="steps", eval_steps=200,
    save_strategy="steps", save_steps=200,
    logging_strategy="steps", logging_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,              # 1 epoch for speed
    learning_rate=5e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()  # shows tqdm progress bars by default citeturn0search0
```

### 6. Save & Test

```python
model.save_pretrained("recipe_model")
tokenizer.save_pretrained("recipe_model")

def gen(ings, k=2):
    inps = tokenizer(f"Suggest a recipe given ingredients: {ings}.",
                      return_tensors="pt").to(device)
    outs = model.generate(**inps, max_length=200, num_beams=k, num_return_sequences=k)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outs]

for r in gen("tomato, basil, garlic", k=2):
    print(r, "\n---")
```
> Fine‑tuning on your specific dataset yields better alignment with assignment specs citeturn2search3turn2search5.

---

### References
1. Pipelines overview – **Hugging Face** citeturn0search0  
2. Text2Text pipeline docs – **Hugging Face** citeturn0search1  
3. Model list for text2text – **Hugging  Face** citeturn0search2  
4. `text2text-generation` source – **Transformers GitHub** citeturn0search4  
5. `t5-recipe-generation` model card – **Hugging  Face** citeturn1search0  
6. Flax Community demo – **m3hrdadfi’s server.py** citeturn1search5  
7. Recipe‑NLG mention – **Hugging  Face Forum** citeturn1search4  
8. Fine‑tuning guide – **NLPlanet** citeturn2search0  
9. T5‑small fine‑tuning tutorial – **Medium** citeturn2search7  
10. Fine‑tuning best practices – **Hugging Face Forum** citeturn2search6  

These two schemes give you both “instant” and “custom” pathways to satisfy your assignment, complete with code you can paste into Colab and run end‑to‑end.
