"""
Model Training Module for Legal Awareness System
Fine-tunes InLegalBERT on merged legal QA dataset
"""
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers import util

class LegalModelTrainer:
    def __init__(self, data_path, model_name="law-ai/InLegalBERT"):
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.model = None
        self.train_df = None
        self.test_df = None
        
    def load_data(self):
        """Load the prepared legal QA dataset"""
        print(f"📂 Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(df)} QA pairs")
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.train_df, self.test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['category']
        )
        print(f"📊 Training samples: {len(self.train_df)}")
        print(f"📊 Testing samples: {len(self.test_df)}")
        return self.train_df, self.test_df
    
    def load_base_model(self):
        """Load the base InLegalBERT model"""
        print(f"🔧 Loading base model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("✅ Base model loaded successfully")
        return self.model
    
    def evaluate_base_model(self):
        """Evaluate base model performance before fine-tuning"""
        print("\n" + "="*60)
        print("📊 Evaluating BASE MODEL Performance")
        print("="*60)
        
        # Encode embeddings
        train_embeddings = self.model.encode(
            self.train_df["question"].tolist(), 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        test_embeddings = self.model.encode(
            self.test_df["question"].tolist(), 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        train_answer_embeddings = self.model.encode(
            self.train_df["answer"].tolist(), 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        test_answer_embeddings = self.model.encode(
            self.test_df["answer"].tolist(), 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        
        # Evaluate with semantic matching
        semantic_correct = 0
        top_k = 3
        threshold = 0.65
        
        for i, test_emb in enumerate(test_embeddings):
            cosine_scores = util.cos_sim(test_emb, train_embeddings)[0]
            top_k_idx = torch.topk(cosine_scores, k=top_k).indices.tolist()
            correct_answer_emb = test_answer_embeddings[i]
            
            match_found = False
            for idx in top_k_idx:
                retrieved_answer_emb = train_answer_embeddings[idx]
                sim = util.cos_sim(correct_answer_emb, retrieved_answer_emb).item()
                if sim >= threshold:
                    match_found = True
                    break
            
            if match_found:
                semantic_correct += 1
        
        base_accuracy = semantic_correct / len(self.test_df)
        print(f"✅ Base Model Semantic Top-{top_k} Accuracy: {base_accuracy:.2%}")
        return base_accuracy
    
    def prepare_training_examples(self):
        """Prepare training examples for fine-tuning"""
        print("\n🔧 Preparing training examples...")
        train_examples = [
            InputExample(texts=[row['question'], row['answer']])
            for _, row in self.train_df.iterrows()
        ]
        print(f"✅ Created {len(train_examples)} training examples")
        return train_examples
    
    def fine_tune_model(self, train_examples, epochs=3, batch_size=16, output_dir=None):
        """Fine-tune the model on legal QA data"""
        print("\n" + "="*60)
        print("🚀 Starting Model Fine-Tuning")
        print("="*60)
        
        # Disable wandb
        os.environ["WANDB_DISABLED"] = "true"
        
        # Create dataloader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Calculate warmup steps
        warmup_steps = int(len(train_dataloader) * epochs * 0.1)
        
        print(f"📊 Training Configuration:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch Size: {batch_size}")
        print(f"   - Warmup Steps: {warmup_steps}")
        print(f"   - Total Batches: {len(train_dataloader)}")
        
        # Fine-tune
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
        )
        
        # Save model
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "models" / "fine_tuned_legal_bert"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir))
        print(f"\n💾 Fine-tuned model saved to: {output_dir}")
        
        return output_dir
    
    def evaluate_finetuned_model(self):
        """Evaluate fine-tuned model performance"""
        print("\n" + "="*60)
        print("📊 Evaluating FINE-TUNED MODEL Performance")
        print("="*60)
        
        # Encode embeddings with fine-tuned model
        train_embeddings = self.model.encode(
            self.train_df["question"].tolist(), 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        test_embeddings = self.model.encode(
            self.test_df["question"].tolist(), 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        train_answer_embeddings = self.model.encode(
            self.train_df["answer"].tolist(), 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        test_answer_embeddings = self.model.encode(
            self.test_df["answer"].tolist(), 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        
        # Evaluate with semantic matching
        semantic_correct = 0
        top_k = 3
        threshold = 0.65
        
        for i, test_emb in enumerate(test_embeddings):
            cosine_scores = util.cos_sim(test_emb, train_embeddings)[0]
            top_k_idx = torch.topk(cosine_scores, k=top_k).indices.tolist()
            correct_answer_emb = test_answer_embeddings[i]
            
            match_found = False
            for idx in top_k_idx:
                retrieved_answer_emb = train_answer_embeddings[idx]
                sim = util.cos_sim(correct_answer_emb, retrieved_answer_emb).item()
                if sim >= threshold:
                    match_found = True
                    break
            
            if match_found:
                semantic_correct += 1
        
        finetuned_accuracy = semantic_correct / len(self.test_df)
        print(f"✅ Fine-tuned Model Semantic Top-{top_k} Accuracy: {finetuned_accuracy:.2%}")
        return finetuned_accuracy
    
    def test_criminal_queries(self):
        """Test model on criminal law queries (like stabbing, murder)"""
        print("\n" + "="*60)
        print("🔪 Testing Criminal Law Queries")
        print("="*60)
        
        # Define test queries
        criminal_queries = [
            "What if I stabbed someone with a knife?",
            "What is the punishment for attempt to murder?",
            "What is Section 307 IPC?",
            "What happens if someone tries to kill another person?",
            "What is the penalty for grievous hurt?"
        ]
        
        # Encode train data
        train_embeddings = self.model.encode(
            self.train_df["question"].tolist(), 
            convert_to_tensor=True
        )
        
        for query in criminal_queries:
            print(f"\n❓ Query: {query}")
            
            # Encode query
            query_emb = self.model.encode([query], convert_to_tensor=True)
            
            # Find best match
            cosine_scores = util.cos_sim(query_emb, train_embeddings)[0]
            best_idx = int(torch.argmax(cosine_scores))
            score = cosine_scores[best_idx].item()
            
            # Get matched QA
            matched_q = self.train_df.iloc[best_idx]['question']
            matched_a = self.train_df.iloc[best_idx]['answer']
            category = self.train_df.iloc[best_idx]['category']
            
            print(f"✅ Match Score: {score:.3f}")
            print(f"📌 Category: {category}")
            print(f"🔍 Matched Question: {matched_q}")
            print(f"💡 Answer: {matched_a[:200]}...")
    
    def run_complete_pipeline(self, output_dir=None):
        """Run the complete training pipeline"""
        print("\n" + "="*70)
        print("🎯 LEGAL AWARENESS MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Load data
        df = self.load_data()
        
        # Split data
        self.split_data(df)
        
        # Load base model
        self.load_base_model()
        
        # Evaluate base model
        base_acc = self.evaluate_base_model()
        
        # Prepare training examples
        train_examples = self.prepare_training_examples()
        
        # Fine-tune model
        model_path = self.fine_tune_model(train_examples, output_dir=output_dir)
        
        # Evaluate fine-tuned model
        finetuned_acc = self.evaluate_finetuned_model()
        
        # Test on criminal queries
        self.test_criminal_queries()
        
        # Final summary
        print("\n" + "="*70)
        print("📊 TRAINING SUMMARY")
        print("="*70)
        print(f"📈 Base Model Accuracy: {base_acc:.2%}")
        print(f"📈 Fine-tuned Model Accuracy: {finetuned_acc:.2%}")
        print(f"📈 Improvement: {(finetuned_acc - base_acc):.2%}")
        print(f"💾 Model saved to: {model_path}")
        print("\n✅ Training pipeline completed successfully!")
        
        return model_path


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "final_merged_legal_qa.csv"
    output_dir = base_dir / "models" / "fine_tuned_legal_bert"
    
    # Initialize trainer
    trainer = LegalModelTrainer(data_path)
    
    # Run complete pipeline
    model_path = trainer.run_complete_pipeline(output_dir=output_dir)
