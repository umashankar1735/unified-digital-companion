# scripts/4_train_t5.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "faq_qa.jsonl"

# Check if data file exists
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

print(f"Loading data from: {DATA_FILE}")

ds = load_dataset('json', data_files=str(DATA_FILE), split='train')
print(f"Loaded {len(ds)} examples")

# create train/dev split
ds = ds.train_test_split(test_size=0.2, seed=42)
train_ds = ds['train']
eval_ds = ds['test']
print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

MODEL = "t5-small"
print(f"Loading model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

max_input_length = 256
max_target_length = 128

def preprocess(examples):
    # Input: just the question with a prefix
    inputs = ["Answer this question: " + q for q in examples['question']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='max_length')
    
    # Target: the answer
    labels = tokenizer(examples['answer'], max_length=max_target_length, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
output_dir = ROOT / "models" / "t5_faq"
output_dir.mkdir(parents=True, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(output_dir),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    eval_strategy="epoch",
    logging_steps=50,
    num_train_epochs=3,
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()
print("Training completed!")

final_model_dir = ROOT / "models" / "t5_faq_final"
trainer.save_model(str(final_model_dir))
print(f"Model saved to: {final_model_dir}")