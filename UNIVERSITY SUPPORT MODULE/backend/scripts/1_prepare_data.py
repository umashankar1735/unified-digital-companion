# scripts/1_prepare_data.py
import pandas as pd
import re
import os
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_IN = ROOT / "data" / "Campus_Knowledge_Dataset.xlsx"
OUT_DIR = ROOT / "data" 
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # replace mention of guide/contact rule to Student Affairs
    s = re.sub(r"contact any guide", "contact Student Affairs / Student Council", s, flags=re.I)
    # more substitutions if needed
    return s

def main():
    print(f"Reading Excel file: {DATA_IN}")
    if not DATA_IN.exists():
        raise FileNotFoundError(f"Excel file not found: {DATA_IN}")
    
    df = pd.read_excel(DATA_IN)
    print(f"Loaded DataFrame with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Ensure columns: Question;Answer;Category;Keywords - they should be in columns already
    # Standardize names if different
    if 'Question' not in df.columns:
        raise SystemExit("Expected 'Question' column in Excel")
    df = df[['Question','Answer','Category','Keywords']].fillna("")
    df['Question'] = df['Question'].apply(normalize_text)
    df['Answer'] = df['Answer'].apply(normalize_text)
    df['Category'] = df['Category'].apply(lambda x: x.strip() if isinstance(x,str) else "")
    df['id'] = df.index + 1

    # Save CSV for indexing
    out_csv = OUT_DIR / "passages.csv"
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"Successfully wrote {len(df)} rows to {out_csv}")

    # Save JSONL for seq2seq fine-tuning (question+answer pairs)
    out_jsonl = OUT_DIR / "faq_qa.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            obj = {"id": str(row['id']),
                   "question": row['Question'],
                   "answer": row['Answer'],
                   "category": row['Category'],
                   "keywords": row['Keywords']}
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Successfully wrote {len(df)} entries to {out_jsonl}")
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        raise