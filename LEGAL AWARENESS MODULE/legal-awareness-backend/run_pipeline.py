"""
Main Script to Run Complete Legal Awareness Pipeline
1. Data Preparation (merge datasets)
2. Model Training (fine-tune InLegalBERT)
3. Model Testing (evaluate and test criminal queries)
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_preparation import DataPreparation
from src.train_model import LegalModelTrainer

def main():
    print("\n" + "="*70)
    print("⚖️  LEGAL AWARENESS SYSTEM - COMPLETE PIPELINE")
    print("="*70)
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    print("\n📋 Pipeline Steps:")
    print("  1. Data Preparation (merge IPC-QA + IndicLegalQA)")
    print("  2. Model Training (fine-tune InLegalBERT)")
    print("  3. Model Evaluation")
    print("  4. Criminal Query Testing")
    
    # Step 1: Data Preparation
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    prep = DataPreparation(data_dir)
    result = prep.prepare_training_data()
    
    if result is None:
        print("❌ Data preparation failed!")
        return False
    
    merged_df, data_path = result
    
    # Step 2: Model Training
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    output_dir = models_dir / "fine_tuned_legal_bert"
    trainer = LegalModelTrainer(data_path)
    
    try:
        model_path = trainer.run_complete_pipeline(output_dir=output_dir)
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Dataset: {len(merged_df)} QA pairs")
        print(f"💾 Model: {model_path}")
        print(f"🚀 Ready to start the API server!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
