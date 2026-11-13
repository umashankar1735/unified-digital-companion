"""
Data Preparation Module for Legal Awareness System
Merges IPC-QA, IndicLegalQA, and other legal datasets
"""
import pandas as pd
import os
from pathlib import Path

class DataPreparation:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
    def load_datasets(self):
        """Load all available datasets"""
        datasets = {}
        
        # Load IPC QA Dataset (Criminal Law - High Priority)
        ipc_path = self.data_dir / "ipc_qa_50.csv"
        if ipc_path.exists():
            ipc_df = pd.read_csv(ipc_path)
            # Standardize columns
            if 'section' in ipc_df.columns:
                ipc_df = ipc_df.rename(columns={'section': 'source'})
            datasets['ipc'] = ipc_df
            print(f"✅ Loaded IPC-QA: {len(ipc_df)} entries")
        
        # Load IndicLegalQA Dataset
        indic_path = self.data_dir / "IndicLegalQA Dataset_10K_Revised.csv"
        if indic_path.exists():
            indic_df = pd.read_csv(indic_path)
            datasets['indic'] = indic_df
            print(f"✅ Loaded IndicLegalQA: {len(indic_df)} entries")
        
        # Load Ace Attorney Dataset (if available)
        ace_path = self.data_dir / "ace_attorney_qa_dataset.csv"
        if ace_path.exists():
            ace_df = pd.read_csv(ace_path)
            datasets['ace'] = ace_df
            print(f"✅ Loaded Ace Attorney QA: {len(ace_df)} entries")
        
        # Load Procedural QA Dataset (How-to guides)
        procedural_path = self.data_dir / "procedural_qa_enhanced.csv"
        if procedural_path.exists():
            procedural_df = pd.read_csv(procedural_path)
            datasets['procedural'] = procedural_df
            print(f"✅ Loaded Procedural QA: {len(procedural_df)} entries")
        
        # Load existing merged dataset (if any)
        merged_path = self.data_dir / "merged_qa_dataset.csv"
        if merged_path.exists():
            merged_df = pd.read_csv(merged_path)
            datasets['existing_merged'] = merged_df
            print(f"ℹ️  Found existing merged dataset: {len(merged_df)} entries")
        
        return datasets
    
    def standardize_columns(self, df, dataset_name):
        """Standardize column names across datasets"""
        # Required columns: question, answer, category (optional), keywords (optional)
        
        # Ensure question and answer columns exist
        if 'question' not in df.columns or 'answer' not in df.columns:
            print(f"⚠️  Warning: {dataset_name} missing required columns")
            return None
        
        # Add missing optional columns
        if 'category' not in df.columns:
            if dataset_name == 'ipc':
                df['category'] = 'Criminal Law - IPC'
            else:
                df['category'] = 'General Legal'
        
        if 'keywords' not in df.columns:
            df['keywords'] = ''
        
        if 'source' not in df.columns:
            df['source'] = dataset_name
        
        # Keep only relevant columns
        return df[['question', 'answer', 'category', 'keywords', 'source']]
    
    def merge_datasets(self, datasets, prioritize_ipc=True):
        """Merge all datasets with IPC-QA given priority"""
        merged_dfs = []
        
        # Prioritize IPC-QA dataset (criminal law queries)
        if prioritize_ipc and 'ipc' in datasets:
            ipc_standardized = self.standardize_columns(datasets['ipc'], 'ipc')
            if ipc_standardized is not None:
                merged_dfs.append(ipc_standardized)
                print(f"✅ Added IPC-QA (priority): {len(ipc_standardized)} entries")
        
        # Add other datasets
        for name, df in datasets.items():
            if name == 'ipc' or name == 'existing_merged':
                continue
            standardized = self.standardize_columns(df, name)
            if standardized is not None:
                merged_dfs.append(standardized)
                print(f"✅ Added {name}: {len(standardized)} entries")
        
        # Merge all dataframes
        if merged_dfs:
            final_df = pd.concat(merged_dfs, ignore_index=True)
            
            # Remove duplicates based on question similarity
            final_df = final_df.drop_duplicates(subset=['question'], keep='first')
            
            print(f"\n✅ Final merged dataset: {len(final_df)} entries")
            print(f"📊 Category distribution:")
            print(final_df['category'].value_counts())
            
            return final_df
        else:
            print("❌ No valid datasets to merge")
            return None
    
    def add_safety_filters(self, df):
        """Add safety keywords for sensitive queries"""
        # Criminal keywords that need careful handling
        criminal_keywords = [
            'murder', 'kill', 'stab', 'knife', 'assault', 'rape', 
            'violence', 'weapon', 'death', 'homicide', 'grievous hurt'
        ]
        
        # Add safety flag
        df['requires_safety_response'] = df.apply(
            lambda row: any(kw in row['question'].lower() or kw in str(row['keywords']).lower() 
                          for kw in criminal_keywords),
            axis=1
        )
        
        safety_count = df['requires_safety_response'].sum()
        print(f"\n🛡️  Safety flags added: {safety_count} entries marked for careful handling")
        
        return df
    
    def save_merged_dataset(self, df, output_filename='final_merged_legal_qa.csv'):
        """Save the merged dataset"""
        output_path = self.data_dir / output_filename
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved merged dataset to: {output_path}")
        return output_path
    
    def prepare_training_data(self):
        """Complete pipeline: load, merge, filter, save"""
        print("=" * 60)
        print("🔧 Legal Awareness Data Preparation Pipeline")
        print("=" * 60)
        
        # Load all datasets
        datasets = self.load_datasets()
        
        if not datasets:
            print("❌ No datasets found!")
            return None
        
        # Merge datasets
        merged_df = self.merge_datasets(datasets, prioritize_ipc=True)
        
        if merged_df is None:
            return None
        
        # Add safety filters
        merged_df = self.add_safety_filters(merged_df)
        
        # Save merged dataset
        output_path = self.save_merged_dataset(merged_df)
        
        print("\n✅ Data preparation complete!")
        return merged_df, output_path


if __name__ == "__main__":
    # Get the data directory path
    base_dir = Path(__file__).parent.parent / "data"
    
    # Initialize and run preparation
    prep = DataPreparation(base_dir)
    result = prep.prepare_training_data()
    
    if result:
        df, path = result
        print(f"\n📈 Dataset ready for training!")
        print(f"📊 Total entries: {len(df)}")
        print(f"📁 Location: {path}")
