#!/usr/bin/env python3
"""
University Support Module - Performance Evaluation Script
Generates metrics shown in the paper: Recall@K, MRR, MAP, Accuracy, Confidence, Response Time
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import json

# Add scripts to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from _utils import hybrid_retrieve, compute_confidence
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle

class UniversitySupportEvaluator:
    def __init__(self):
        self.ROOT = Path(__file__).resolve().parents[1]
        self.indices_dir = self.ROOT / "indices"
        self.data_dir = self.ROOT / "data"
        
        print("\n" + "="*80)
        print("🎓 UNIVERSITY SUPPORT MODULE - EVALUATION")
        print("="*80)
        
        # Load data
        print("\n📂 Loading components...")
        self.passages_df = pd.read_csv(self.data_dir / "passages.csv")
        self.questions = self.passages_df['Question'].tolist()
        self.answers = self.passages_df['Answer'].tolist()
        
        # Load model
        print("🔧 Loading sentence transformer...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load indices
        print("🔧 Loading search indices...")
        with open(self.indices_dir / "bm25.pkl", "rb") as f:
            bm25_obj = pickle.load(f)
            self.bm25 = bm25_obj["bm25"]
            self.passages = bm25_obj["passages"]
        
        self.faiss_index = faiss.read_index(str(self.indices_dir / "faiss_index.bin"))
        
        # Load reranker
        try:
            print("🔧 Loading cross-encoder reranker...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.has_reranker = True
        except:
            print("⚠️  Reranker not available")
            self.has_reranker = False
        
        print(f"✅ Loaded {len(self.questions)} QA pairs\n")
    
    def retrieve_with_reranker(self, query: str, k: int = 10) -> list:
        """Retrieve with reranking"""
        # Get initial candidates (2x K)
        candidates = hybrid_retrieve(query, k=k*2)
        
        if not self.has_reranker or not candidates:
            return candidates[:k]
        
        # Rerank
        pairs = [[query, c['passage']] for c in candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort by reranker scores
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [c for c, _ in scored[:k]]
    
    def calculate_recall_at_k(self, test_size: int = 500, k_values: list = [1, 3, 5, 10], 
                               use_reranker: bool = False) -> dict:
        """Calculate Recall@K metrics"""
        print(f"\n{'='*80}")
        print(f"📊 Calculating Recall@K ({'With Reranker' if use_reranker else 'Baseline'})")
        print(f"{'='*80}")
        
        np.random.seed(42)
        test_indices = np.random.choice(len(self.questions), min(test_size, len(self.questions)), replace=False)
        
        recall_scores = {k: [] for k in k_values}
        all_times = []
        all_confidences = []
        
        for idx in test_indices:
            query = self.questions[idx]
            correct_answer = self.answers[idx]
            
            start_time = time.time()
            
            # Retrieve
            if use_reranker:
                results = self.retrieve_with_reranker(query, k=max(k_values))
            else:
                results = hybrid_retrieve(query, k=max(k_values))
            
            retrieval_time = (time.time() - start_time) * 1000
            all_times.append(retrieval_time)
            
            # Get confidence
            conf = compute_confidence(query, top_k=1)
            all_confidences.append(conf)
            
            # Check if correct answer is in top-K
            retrieved_passages = [r['passage'] for r in results]
            
            for k in k_values:
                top_k_passages = retrieved_passages[:k]
                # Check if correct answer is in top-K
                found = any(correct_answer.strip() in passage for passage in top_k_passages)
                recall_scores[k].append(1 if found else 0)
        
        metrics = {
            f'Recall@{k}': np.mean(recall_scores[k]) for k in k_values
        }
        metrics['Avg_Response_Time_ms'] = np.mean(all_times)
        metrics['Avg_Confidence'] = np.mean(all_confidences)
        
        # Calculate MRR
        mrr_scores = []
        for idx in test_indices:
            query = self.questions[idx]
            correct_answer = self.answers[idx]
            
            if use_reranker:
                results = self.retrieve_with_reranker(query, k=10)
            else:
                results = hybrid_retrieve(query, k=10)
            
            retrieved_passages = [r['passage'] for r in results]
            
            # Find rank of correct answer
            for rank, passage in enumerate(retrieved_passages, 1):
                if correct_answer.strip() in passage:
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)
        
        metrics['MRR'] = np.mean(mrr_scores)
        
        # Calculate MAP
        map_scores = []
        for idx in test_indices:
            query = self.questions[idx]
            correct_answer = self.answers[idx]
            
            if use_reranker:
                results = self.retrieve_with_reranker(query, k=10)
            else:
                results = hybrid_retrieve(query, k=10)
            
            retrieved_passages = [r['passage'] for r in results]
            
            for rank, passage in enumerate(retrieved_passages, 1):
                if correct_answer.strip() in passage:
                    map_scores.append(1.0 / rank)
                    break
            else:
                map_scores.append(0.0)
        
        metrics['MAP'] = np.mean(map_scores)
        
        return metrics
    
    def evaluate_answer_accuracy(self, test_size: int = 200, use_reranker: bool = False) -> float:
        """Evaluate answer accuracy"""
        np.random.seed(42)
        test_indices = np.random.choice(len(self.questions), min(test_size, len(self.questions)), replace=False)
        
        correct = 0
        for idx in test_indices:
            query = self.questions[idx]
            correct_answer = self.answers[idx]
            
            if use_reranker:
                results = self.retrieve_with_reranker(query, k=1)
            else:
                results = hybrid_retrieve(query, k=1)
            
            if results and correct_answer.strip() in results[0]['passage']:
                correct += 1
        
        return correct / len(test_indices)
    
    def calculate_escalation_rate(self, confidence_threshold: float = 0.45) -> float:
        """Calculate HITL escalation rate"""
        confidences = []
        
        for query in self.questions[:500]:
            conf = compute_confidence(query, top_k=1)
            confidences.append(conf)
        
        below_threshold = sum(1 for c in confidences if c < confidence_threshold)
        return below_threshold / len(confidences)
    
    def run_complete_evaluation(self) -> dict:
        """Run complete evaluation"""
        print("\n" + "="*80)
        print("🚀 RUNNING COMPLETE EVALUATION")
        print("="*80)
        
        results = {
            'baseline': {},
            'with_reranker': {}
        }
        
        # Baseline
        print("\n📊 Evaluating BASELINE system...")
        baseline_metrics = self.calculate_recall_at_k(test_size=500, use_reranker=False)
        baseline_metrics['Answer_Accuracy'] = self.evaluate_answer_accuracy(test_size=200, use_reranker=False)
        baseline_metrics['Escalation_Rate'] = self.calculate_escalation_rate()
        results['baseline'] = baseline_metrics
        
        # With reranker
        if self.has_reranker:
            print("\n📊 Evaluating WITH RERANKER...")
            reranker_metrics = self.calculate_recall_at_k(test_size=500, use_reranker=True)
            reranker_metrics['Answer_Accuracy'] = self.evaluate_answer_accuracy(test_size=200, use_reranker=True)
            reranker_metrics['Escalation_Rate'] = self.calculate_escalation_rate()
            results['with_reranker'] = reranker_metrics
        
        return results
    
    def print_results(self, results: dict):
        """Print formatted results"""
        print("\n" + "="*80)
        print("📊 EVALUATION RESULTS - UNIVERSITY SUPPORT MODULE")
        print("="*80)
        
        metrics = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'MRR', 'MAP', 
                   'Answer_Accuracy', 'Avg_Confidence', 'Escalation_Rate', 'Avg_Response_Time_ms']
        
        print(f"\n{'Metric':<25} {'Baseline':<15} {'With Reranker':<15} {'Improvement':<15}")
        print("-" * 80)
        
        for metric in metrics:
            baseline_val = results['baseline'].get(metric, 0)
            reranker_val = results['with_reranker'].get(metric, 0) if self.has_reranker else 0
            
            if metric == 'Avg_Response_Time_ms':
                if baseline_val > 0:
                    change = ((reranker_val - baseline_val) / baseline_val) * 100
                    print(f"{metric:<25} {baseline_val:<15.2f} {reranker_val:<15.2f} {change:>+.1f}%")
                else:
                    print(f"{metric:<25} {baseline_val:<15.2f} {reranker_val:<15.2f} {'N/A':<15}")
            else:
                if baseline_val > 0 and reranker_val > 0:
                    improvement = ((reranker_val - baseline_val) / baseline_val) * 100
                    print(f"{metric:<25} {baseline_val:<15.4f} {reranker_val:<15.4f} {improvement:>+.1f}%")
                else:
                    print(f"{metric:<25} {baseline_val:<15.4f} {reranker_val:<15.4f} {'N/A':<15}")
        
        print("=" * 80)
        
        # Convert numpy types to Python types for JSON serialization
        results_json = {
            'baseline': {k: float(v) for k, v in results['baseline'].items()},
            'with_reranker': {k: float(v) for k, v in results['with_reranker'].items()}
        }
        
        # Save to JSON
        output_file = self.ROOT / "evaluation_results_university.json"
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    evaluator = UniversitySupportEvaluator()
    results = evaluator.run_complete_evaluation()
    evaluator.print_results(results)
