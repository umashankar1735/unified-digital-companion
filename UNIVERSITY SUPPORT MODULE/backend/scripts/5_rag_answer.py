# scripts/5_rag_answer.py
import os
# Set threading environment variables BEFORE importing any ML libraries
# This prevents mutex blocking issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import json
import sys

# Add the scripts directory to the path so we can import _utils
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

try:
    from _utils import hybrid_retrieve, compute_confidence
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure you've run 2_build_indices.py first to create the search indices")
    sys.exit(1)

# Import logging function - use module name instead of scripts prefix
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("logging_module", script_dir / "7_logging.py")
    logging_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logging_module)
    log_escalation = logging_module.log_escalation
except Exception as e:
    log_escalation = None

# Import reranker
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except ImportError:
    RERANKER_AVAILABLE = False
    reranker_model = None

ROOT = Path(__file__).resolve().parents[1]

# Try to find the trained model, check both possible locations
MODEL_DIR_FINAL = ROOT / "models" / "t5_faq_final"
MODEL_DIR = ROOT / "models" / "t5_faq"

if MODEL_DIR_FINAL.exists():
    model_name = str(MODEL_DIR_FINAL)
elif MODEL_DIR.exists():
    model_name = str(MODEL_DIR)
else:
    model_name = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def rerank_passages(question, candidates):
    """Rerank retrieved passages using CrossEncoder for better relevance"""
    if not RERANKER_AVAILABLE or not candidates:
        return candidates
    
    try:
        # Create question-passage pairs for reranking
        pairs = [[question, c['passage']] for c in candidates]
        scores = reranker_model.predict(pairs)
        
        # Sort by scores (highest first)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        reranked_candidates = [r[0] for r in ranked]
        
        return reranked_candidates
    except Exception as e:
        return candidates  # Silently fall back to original order

def build_context_snippet(retrieved):
    # Take top 2 most relevant passages to avoid overwhelming the model
    if not retrieved:
        return ""
    
    texts = []
    for i, r in enumerate(retrieved[:2]):
        passage = r['passage'].strip()
        if passage:
            texts.append(passage)
    
    # Join with clear separators
    ctx = " | ".join(texts)
    return ctx

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

def analyze_confidence_distribution(sample_questions=None):
    """Analyze confidence scores for different questions to help tune threshold"""
    if sample_questions is None:
        sample_questions = [
            "Where is the library located?",
            "How can I reach college from vijayawada?", 
            "What are the gym timings?",
            "Who should I contact for IT support?",
            "What is the meaning of life?",  # Should have low confidence
            "How do I become a unicorn?",    # Should have low confidence
        ]
    
    print("=== Confidence Analysis ===")
    confidences = []
    for q in sample_questions:
        try:
            conf = compute_confidence(q, top_k=1)
            confidences.append(conf)
            print(f"Q: {q[:50]:<50} | Confidence: {conf:.3f}")
        except Exception as e:
            print(f"Q: {q[:50]:<50} | Error: {e}")
    
    if confidences:
        import statistics
        print(f"\nConfidence Statistics:")
        print(f"Min: {min(confidences):.3f}")
        print(f"Max: {max(confidences):.3f}")
        print(f"Mean: {statistics.mean(confidences):.3f}")
        print(f"Median: {statistics.median(confidences):.3f}")
        print("\nSuggested threshold: Around 0.4-0.5 (adjust based on your needs)")
    print("=" * 70)

def generate_answer(question, k=5):
    try:
        # Enhance the query based on question type for better retrieval
        enhanced_question = question
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['where', 'location', 'located', 'college']):
            enhanced_question = f"{question} campus amaravati amrita location"
        elif any(word in question_lower for word in ['how', 'reach', 'transport', 'bus', 'travel', 'timing']):
            enhanced_question = f"{question} bus transport route boarding points vijayawada guntur timing schedule"
        elif any(word in question_lower for word in ['when', 'time', 'schedule', 'timing']):
            enhanced_question = f"{question} time schedule hours timing am pm"
        elif any(word in question_lower for word in ['who', 'contact', 'email', 'head']):
            enhanced_question = f"{question} contact email department affairs sujatha kavitha"
        elif any(word in question_lower for word in ['participate', 'sports', 'event', 'competition']):
            enhanced_question = f"{question} sports participate register competition tournament physical education"
        elif any(word in question_lower for word in ['lose', 'lost', 'happens', 'policy']):
            enhanced_question = f"{question} policy procedure rules lost student affairs"
        
        # Retrieve more candidates for reranking (get k*2 to have more options)
        initial_k = min(k * 2, 10) if RERANKER_AVAILABLE else k
        retrieved = hybrid_retrieve(enhanced_question, k=initial_k)
        
        # Rerank the retrieved passages for better relevance
        if RERANKER_AVAILABLE and len(retrieved) > 1:
            retrieved = rerank_passages(question, retrieved)
            # Take top k after reranking
            retrieved = retrieved[:k]
        else:
            retrieved = retrieved[:k]
        
        # Retrieval completed silently for clean demo
        
        ctx = build_context_snippet(retrieved)
        
        # Compute confidence first for return value
        try:
            conf = compute_confidence(question, top_k=1)
        except:
            conf = 0.0
        
        # Check confidence threshold early
        LOW_CONFIDENCE_THRESHOLD = 0.45
        if conf < LOW_CONFIDENCE_THRESHOLD:
            # Log the low-confidence query for human review
            snippet = retrieved[0]['passage'][:300] if retrieved else ""
            if log_escalation:
                log_escalation(question, conf, snippet)
            
            # Generate appropriate fallback response
            fallback_answer = generate_fallback_response(question)
            return fallback_answer, retrieved, conf
        
        # If we have context, use it; otherwise just use the base model
        if ctx.strip():
            # Use a more explicit format for better results
            prompt = f"question: {question} context: {ctx}"
        else:
            prompt = f"question: {question}"
            
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate with better parameters
        out = model.generate(
            **inputs, 
            max_length=128, 
            num_beams=4, 
            early_stopping=True,
            do_sample=False
        )
        answer = tokenizer.decode(out[0], skip_special_tokens=True)
        
        # If answer is too short, incomplete, or just fragments, use extractive approach
        if (len(answer.split()) < 5 or 
            answer.strip().lower() in ['true', 'false', 'yes', 'no'] or
            not answer.strip().endswith(('.', '!', '?')) or
            len(answer.strip()) < 20):
            
            # Use extractive answer from the most relevant passage
            if retrieved and len(retrieved) > 0:
                best_passage = retrieved[0]['passage']
                
                # For specific question types, extract relevant information
                question_lower = question.lower()
                
                if any(word in question_lower for word in ['who', 'head', 'contact']):
                    # For "who" questions, look for names and titles
                    if 'sujatha' in best_passage.lower():
                        answer = "Mrs. Sujatha is the head of Student Affairs."
                    elif 'kavitha mutyala' in best_passage.lower():
                        answer = "Dr. Kavitha Mutyala heads the Department of Physical Education."
                    else:
                        answer = best_passage
                        
                elif any(word in question_lower for word in ['timing', 'time', 'when', 'schedule']):
                    # For timing questions, extract time information
                    answer = best_passage
                    
                elif any(word in question_lower for word in ['how', 'participate', 'register']):
                    # For "how to" questions, provide the full procedure
                    answer = best_passage
                    
                elif any(word in question_lower for word in ['what happens', 'lose', 'lost']):
                    # For "what happens" questions
                    answer = best_passage
                    
                else:
                    # Default: use the best passage
                    answer = best_passage
            
        return answer, retrieved, conf
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating the answer.", [], 0.0

if __name__ == "__main__":
    print("Campus Knowledge Assistant")
    print("=" * 40)
    
    while True:
        try:
            q = input("\nQuestion: ")
            if q.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            ans, retrieved, conf = generate_answer(q)
            print(f"Answer: {ans}")
            print(f"Confidence: {conf:.3f}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue