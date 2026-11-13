#!/usr/bin/env python3
"""
Legal Awareness Module - Performance Evaluation Script
Generates metrics shown in the paper: Recall@K, MRR, MAP, Accuracy, Confidence, Response Time
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch
import time
from typing import List, Dict, Tuple
import json

class LegalSystemEvaluator:
    def __init__(self, model_path, data_path):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        
        print("\n" + "="*80)
        print("⚖️  LEGAL AWARENESS SYSTEM - EVALUATION")
        print("="*80)
        
        # Load model and data
        print("\n📂 Loading components...")
        self.model = SentenceTransformer(str(model_path))
        self.qa_data = pd.read_csv(data_path)
        
        # Create embeddings
        print(f"🔧 Creating embeddings for {len(self.qa_data)} QA pairs...")
        self.questions = self.qa_data['question'].tolist()
        self.answers = self.qa_data['answer'].tolist()
        self.question_embeddings = self.model.encode(
            self.questions, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Load reranker for comparison
        try:
            print("🔧 Loading cross-encoder reranker...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.has_reranker = True
        except:
            print("⚠️  Reranker not available - will compare baseline only")
            self.has_reranker = False
        
        print("✅ Initialization complete!\n")
    
    def retrieve_top_k(self, query: str, k: int = 10, use_reranker: bool = False) -> List[Tuple[int, float]]:
        """Retrieve top-K most similar questions"""
        start_time = time.time()
        
        # Semantic search
        query_emb = self.model.encode([query], convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.question_embeddings)[0]
        
        # Get top candidates
        top_k_initial = min(k * 2, len(scores)) if use_reranker else k
        top_indices = torch.topk(scores, k=top_k_initial).indices.tolist()
        top_scores = torch.topk(scores, k=top_k_initial).values.tolist()
        
        # Rerank if enabled
        if use_reranker and self.has_reranker:
            pairs = [[query, self.questions[idx]] for idx in top_indices]
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine and re-sort
            scored_results = list(zip(top_indices, rerank_scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            results = scored_results[:k]
        else:
            results = list(zip(top_indices, top_scores))
        
        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
        return results, retrieval_time
    
    def calculate_recall_at_k(self, test_size: int = 500, k_values: List[int] = [1, 3, 5, 10],
                               use_reranker: bool = False) -> Dict:
        """Calculate Recall@K metrics"""
        print(f"\n{'='*80}")
        print(f"📊 Calculating Recall@K ({'With Reranker' if use_reranker else 'Baseline'})")
        print(f"{'='*80}")
        
        # Sample test queries
        np.random.seed(42)
        test_indices = np.random.choice(len(self.questions), min(test_size, len(self.questions)), replace=False)
        
        recall_scores = {k: [] for k in k_values}
        all_times = []
        all_confidences = []
        
        for idx in test_indices:
            query = self.questions[idx]
            correct_idx = idx
            
            # Retrieve top-K
            results, retrieval_time = self.retrieve_top_k(query, k=max(k_values), use_reranker=use_reranker)
            retrieved_indices = [r[0] for r in results]
            
            all_times.append(retrieval_time)
            if results:
                all_confidences.append(results[0][1])  # Top-1 score
            
            # Calculate recall for each K
            for k in k_values:
                top_k_indices = retrieved_indices[:k]
                recall_scores[k].append(1 if correct_idx in top_k_indices else 0)
        
        # Compute averages
        metrics = {
            f'Recall@{k}': np.mean(recall_scores[k]) for k in k_values
        }
        metrics['Avg_Response_Time_ms'] = np.mean(all_times)
        metrics['Avg_Confidence'] = np.mean(all_confidences)
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr_scores = []
        for idx in test_indices:
            query = self.questions[idx]
            results, _ = self.retrieve_top_k(query, k=10, use_reranker=use_reranker)
            retrieved_indices = [r[0] for r in results]
            
            if idx in retrieved_indices:
                rank = retrieved_indices.index(idx) + 1
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)
        
        metrics['MRR'] = np.mean(mrr_scores)
        
        # Calculate MAP (Mean Average Precision)
        map_scores = []
        for idx in test_indices:
            query = self.questions[idx]
            results, _ = self.retrieve_top_k(query, k=10, use_reranker=use_reranker)
            retrieved_indices = [r[0] for r in results]
            
            if idx in retrieved_indices:
                rank = retrieved_indices.index(idx) + 1
                precision_at_k = 1.0 / rank
                map_scores.append(precision_at_k)
            else:
                map_scores.append(0.0)
        
        metrics['MAP'] = np.mean(map_scores)
        
        return metrics
    
    def evaluate_answer_accuracy(self, test_size: int = 200, use_reranker: bool = False) -> float:
        """Evaluate answer accuracy (top-1 retrieval)"""
        np.random.seed(42)
        test_indices = np.random.choice(len(self.questions), min(test_size, len(self.questions)), replace=False)
        
        correct = 0
        for idx in test_indices:
            query = self.questions[idx]
            results, _ = self.retrieve_top_k(query, k=1, use_reranker=use_reranker)
            
            if results and results[0][0] == idx:
                correct += 1
        
        return correct / len(test_indices)
    
    def calculate_escalation_rate(self, confidence_threshold: float = 0.45) -> float:
        """Calculate HITL escalation rate"""
        confidences = []
        
        for query in self.questions[:500]:
            results, _ = self.retrieve_top_k(query, k=1, use_reranker=False)
            if results:
                confidences.append(results[0][1])
        
        below_threshold = sum(1 for c in confidences if c < confidence_threshold)
        return below_threshold / len(confidences)
    
    def run_complete_evaluation(self) -> Dict:
        """Run complete evaluation and generate paper metrics"""
        print("\n" + "="*80)
        print("🚀 RUNNING COMPLETE EVALUATION")
        print("="*80)
        
        results = {
            'baseline': {},
            'with_reranker': {}
        }
        
        # Baseline evaluation
        print("\n📊 Evaluating BASELINE system...")
        baseline_metrics = self.calculate_recall_at_k(test_size=500, use_reranker=False)
        baseline_metrics['Answer_Accuracy'] = self.evaluate_answer_accuracy(test_size=200, use_reranker=False)
        baseline_metrics['Escalation_Rate'] = self.calculate_escalation_rate()
        results['baseline'] = baseline_metrics
        
        # Reranker evaluation
        if self.has_reranker:
            print("\n📊 Evaluating WITH RERANKER...")
            reranker_metrics = self.calculate_recall_at_k(test_size=500, use_reranker=True)
            reranker_metrics['Answer_Accuracy'] = self.evaluate_answer_accuracy(test_size=200, use_reranker=True)
            reranker_metrics['Escalation_Rate'] = self.calculate_escalation_rate()
            results['with_reranker'] = reranker_metrics
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted results table"""
        print("\n" + "="*80)
        print("📊 EVALUATION RESULTS - LEGAL AWARENESS MODULE")
        print("="*80)
        
        metrics = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'MRR', 'MAP', 
                   'Answer_Accuracy', 'Avg_Confidence', 'Escalation_Rate', 'Avg_Response_Time_ms']
        
        print(f"\n{'Metric':<25} {'Baseline':<15} {'With Reranker':<15} {'Improvement':<15}")
        print("-" * 80)
        
        for metric in metrics:
            baseline_val = results['baseline'].get(metric, 0)
            reranker_val = results['with_reranker'].get(metric, 0) if self.has_reranker else 0
            
            if metric == 'Avg_Response_Time_ms':
                # For response time, calculate increase (not improvement)
                if baseline_val > 0:
                    change = ((reranker_val - baseline_val) / baseline_val) * 100
                    print(f"{metric:<25} {baseline_val:<15.2f} {reranker_val:<15.2f} {change:>+.1f}%")
                else:
                    print(f"{metric:<25} {baseline_val:<15.2f} {reranker_val:<15.2f} {'N/A':<15}")
            else:
                # For other metrics, calculate improvement
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
        output_file = Path(__file__).parent / "evaluation_results_legal.json"
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    # Paths
    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "fine_tuned_legal_bert"
    data_path = base_dir / "data" / "final_merged_legal_qa.csv"
    
    # Run evaluation
    evaluator = LegalSystemEvaluator(str(model_path), str(data_path))
    results = evaluator.run_complete_evaluation()
    evaluator.print_results(results)
