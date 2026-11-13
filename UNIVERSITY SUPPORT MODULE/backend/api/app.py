# api/app.py
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from scripts._utils import hybrid_retrieve, compute_confidence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import uvicorn
import json
import sys
import os
# Add scripts directory to path
scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
sys.path.append(str(scripts_dir))

# Import the logging function dynamically
import importlib.util
spec = importlib.util.spec_from_file_location("logging_7", scripts_dir / "7_logging.py")
logging_7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging_7)
log_escalation = logging_7.log_escalation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Check for fine-tuned model first, then fallback
MODEL_DIR_FINAL = ROOT / "models" / "t5_faq_final"
MODEL_DIR = ROOT / "models" / "t5_faq"

if MODEL_DIR_FINAL.exists():
    model_name = str(MODEL_DIR_FINAL)
    logger.info(f"Using fine-tuned model: {MODEL_DIR_FINAL}")
elif MODEL_DIR.exists():
    model_name = str(MODEL_DIR)
    logger.info(f"Using trained model: {MODEL_DIR}")
else:
    model_name = "t5-small"
    logger.info(f"Using base model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

app = FastAPI(title="Campus Companion Backend")

class Query(BaseModel):
    question: str
    top_k: int = 5

def generate_fallback_response(question):
    """Generate a polite fallback response for out-of-knowledge queries"""
    
    # Analyze question type to provide targeted fallback
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['admission', 'fee', 'tuition', 'payment']):
        return ("I couldn't find specific information about admissions and fees in our campus knowledge base. "
                "Please contact the Admissions Office or Student Affairs at studentaffairs@av.amrita.edu for "
                "accurate details about admission procedures and fee structure.")
    
    elif any(word in question_lower for word in ['course', 'syllabus', 'curriculum', 'subject']):
        return ("I don't have detailed information about specific courses and curriculum in my knowledge base. "
                "Please contact your department coordinator or Student Affairs at studentaffairs@av.amrita.edu "
                "for course-related information.")
    
    elif any(word in question_lower for word in ['grade', 'marks', 'result', 'exam result']):
        return ("I cannot access individual academic records or results. Please check your results through "
                "the AUMS portal or contact your department for grade-related queries.")
    
    elif any(word in question_lower for word in ['job', 'placement', 'career', 'company']):
        return ("For placement and career guidance information, please contact the Placement & Career Guidance Cell "
                "which operates under Student Affairs, or email studentaffairs@av.amrita.edu.")
    
    elif any(word in question_lower for word in ['personal', 'private', 'confidential']):
        return ("I cannot help with personal or confidential matters. Please contact Student Affairs at "
                "studentaffairs@av.amrita.edu or visit the Student Affairs office for personal assistance.")
    
    else:
        # Generic fallback for unknown topics
        return ("I couldn't find that information in our campus knowledge base. For assistance with your query, "
                "please contact Student Affairs at studentaffairs@av.amrita.edu or visit the Student Council "
                "representatives who can help direct you to the right department.")

def generate_answer(question, retrieved):
    ctx = " ".join([r['passage'] for r in retrieved[:3]])
    prompt = f"Context: {ctx} Question: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**inputs, max_length=200, num_beams=3)
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    return answer

@app.post("/ask")
def ask(q: Query):
    retrieved = hybrid_retrieve(q.question, k=q.top_k)
    conf = compute_confidence(q.question, top_k=1)
    
    if conf < 0.45:
        snippet = retrieved[0]['passage'][:300] if retrieved else ""
        log_escalation(q.question, conf, snippet)
        
        # Generate appropriate fallback response
        fallback_answer = generate_fallback_response(q.question)
        return {"answer": fallback_answer, "confidence": conf, "escalated": True}

    answer = generate_answer(q.question, retrieved)
    sources = [{"snippet": r['passage'][:300]} for r in retrieved[:3]]
    return {"answer": answer, "confidence": conf, "sources": sources, "escalated": False}

@app.get("/escalations")
def get_escalations():
    # return last 100 entries
    LOG = ROOT / "logs" / "escalations.csv"
    lines = []
    if LOG.exists():
        with open(LOG, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()[-100:]
    return {"escalations": lines}

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)