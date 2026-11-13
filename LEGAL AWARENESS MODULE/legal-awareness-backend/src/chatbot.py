"""
Legal Chatbot Inference Module
Handles query processing with safety filters for criminal law queries
"""
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple
import re

class LegalChatbot:
    def __init__(self, model_path, data_path):
        """Initialize the Legal Chatbot"""
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.qa_data = None
        self.qa_embeddings = None
        
        # Criminal law keywords requiring safety response
        self.criminal_keywords = [
            'murder', 'kill', 'stab', 'knife', 'assault', 'rape', 
            'violence', 'weapon', 'death', 'homicide', 'hurt',
            'section 302', 'section 307', 'section 304', 'section 326',
            'attempt to murder', 'culpable homicide', 'grievous hurt'
        ]
        
        self.load_model()
        self.load_data()
        self.create_embeddings()
    
    def load_model(self):
        """Load the fine-tuned model"""
        print(f"🔧 Loading model from: {self.model_path}")
        self.model = SentenceTransformer(str(self.model_path))
        print("✅ Model loaded successfully")
    
    def load_data(self):
        """Load the QA dataset"""
        print(f"📂 Loading QA data from: {self.data_path}")
        self.qa_data = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.qa_data)} QA pairs")
    
    def create_embeddings(self):
        """Create embeddings for all questions in the dataset"""
        print("🔧 Creating question embeddings...")
        questions = self.qa_data["question"].tolist()
        self.qa_embeddings = self.model.encode(
            questions, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        print("✅ Embeddings created successfully")
    
    def is_criminal_query(self, query: str) -> bool:
        """Check if query involves criminal law matters"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.criminal_keywords)
    
    def get_safety_disclaimer(self) -> str:
        """Get the safety disclaimer for criminal law queries"""
        return (
            "\n\n⚠️ **LEGAL DISCLAIMER**: "
            "This information is for legal awareness purposes only. "
            "If you are involved in or witness to a criminal matter, "
            "please immediately contact law enforcement (dial 100) or consult a qualified legal professional. "
            "Do not rely on this information for legal advice in actual situations."
        )
    
    def retrieve_answer(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant answers"""
        # First check for exact or near-exact keyword match (for IPC sections, etc.)
        query_lower = query.lower().strip().replace("?", "").replace(".", "")
        
        # Boost exact keyword matches
        keyword_matches = []
        for idx, row in self.qa_data.iterrows():
            question_lower = str(row['question']).lower().strip().replace("?", "").replace(".", "")
            
            # Check for exact match or very close match
            if query_lower == question_lower:
                keyword_matches.append({
                    'question': row['question'],
                    'answer': row['answer'],
                    'category': row['category'],
                    'score': 0.99,  # Very high confidence for exact match
                    'source': row.get('source', 'Unknown')
                })
            # Check for IPC section specific match
            elif "section" in query_lower and "ipc" in query_lower:
                # Extract section number from query
                import re
                query_sections = re.findall(r'\b\d+[a-z]?\b', query_lower)
                question_sections = re.findall(r'\b\d+[a-z]?\b', question_lower)
                
                if query_sections and question_sections and query_sections[0] == question_sections[0]:
                    keyword_matches.append({
                        'question': row['question'],
                        'answer': row['answer'],
                        'category': row['category'],
                        'score': 0.95,  # High confidence for section number match
                        'source': row.get('source', 'Unknown')
                    })
        
        # If we found exact matches, return them first
        if keyword_matches:
            # Still do semantic search for additional results
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, self.qa_embeddings)[0]
            top_k_indices = torch.topk(cosine_scores, k=top_k).indices.tolist()
            top_k_scores = torch.topk(cosine_scores, k=top_k).values.tolist()
            
            semantic_results = []
            for idx, score in zip(top_k_indices, top_k_scores):
                # Skip if already in keyword matches
                if self.qa_data.iloc[idx]['question'] != keyword_matches[0]['question']:
                    semantic_results.append({
                        'question': self.qa_data.iloc[idx]['question'],
                        'answer': self.qa_data.iloc[idx]['answer'],
                        'category': self.qa_data.iloc[idx]['category'],
                        'score': float(score),
                        'source': self.qa_data.iloc[idx].get('source', 'Unknown')
                    })
            
            # Combine: keyword match first, then semantic matches
            return keyword_matches[:1] + semantic_results[:top_k-1]
        
        # No exact match, use semantic search only
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.qa_embeddings)[0]
        top_k_indices = torch.topk(cosine_scores, k=top_k).indices.tolist()
        top_k_scores = torch.topk(cosine_scores, k=top_k).values.tolist()
        
        results = []
        for idx, score in zip(top_k_indices, top_k_scores):
            result = {
                'question': self.qa_data.iloc[idx]['question'],
                'answer': self.qa_data.iloc[idx]['answer'],
                'category': self.qa_data.iloc[idx]['category'],
                'score': float(score),
                'source': self.qa_data.iloc[idx].get('source', 'Unknown')
            }
            results.append(result)
        
        return results
    
    def format_response(self, query: str, results: List[Dict]) -> str:
        """Format the chatbot response"""
        best_match = results[0]
        is_criminal = self.is_criminal_query(query)
        
        # Check if answer already contains formatting (from IPC dataset)
        answer_text = best_match['answer']
        
        # For IPC sections and other well-formatted answers, just show the answer directly
        if best_match['score'] > 0.9 and (
            best_match.get('source') == 'ipc' or 
            ('Section' in answer_text and 'IPC' in str(best_match['question']))
        ):
            # High confidence IPC answer - show directly without extra wrapping
            response = f"{answer_text}\n\n"
            response += f"**Category:** {best_match['category']}\n"
            response += f"**Confidence Score:** {best_match['score']:.2%}\n"
            response += f"**Source:** IPC Dataset\n"
        else:
            # Regular format for case-based or lower confidence answers
            response = f"**Question:** {best_match['question']}\n\n"
            response += f"**Answer:** {answer_text}\n\n"
            response += f"**Category:** {best_match['category']}\n"
            response += f"**Confidence Score:** {best_match['score']:.2%}\n"
            
            if 'source' in best_match and best_match['source'] != 'Unknown':
                response += f"**Source:** {best_match['source']}\n"
        
        # Add safety disclaimer for criminal queries
        if is_criminal:
            response += self.get_safety_disclaimer()
        
        # Add alternative results if confidence is low
        if best_match['score'] < 0.7 and len(results) > 1:
            response += "\n\n**Other relevant results:**\n"
            for i, result in enumerate(results[1:3], 1):
                response += f"\n{i}. {result['question']} (Score: {result['score']:.2%})"
        
        return response
    
    def get_response(self, query: str, top_k: int = 3) -> Dict:
        """Get chatbot response for a query"""
        # Retrieve relevant answers
        results = self.retrieve_answer(query, top_k=top_k)
        
        # Format response
        formatted_response = self.format_response(query, results)
        
        # Create response object
        response = {
            'query': query,
            'response': formatted_response,
            'best_match': results[0],
            'all_matches': results,
            'is_criminal_query': self.is_criminal_query(query)
        }
        
        return response
    
    def interactive_chat(self):
        """Run interactive chat session"""
        print("\n" + "="*70)
        print("⚖️  LEGAL AWARENESS CHATBOT")
        print("="*70)
        print("Ask me any legal question! Type 'exit' to quit.\n")
        
        while True:
            query = input("You: ").strip()
            
            if query.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Thank you for using Legal Awareness Chatbot!")
                break
            
            if not query:
                continue
            
            print("\n🤔 Thinking...")
            response = self.get_response(query)
            
            print("\n" + "="*70)
            print(response['response'])
            print("="*70 + "\n")


class LegalRetrievalSystem:
    """Advanced retrieval system with BM25 + FAISS hybrid approach"""
    def __init__(self, model_path, data_path):
        self.chatbot = LegalChatbot(model_path, data_path)
        
    def search(self, query: str, filters: Dict = None) -> Dict:
        """
        Search with optional filters
        filters: {'category': 'Criminal Law', 'min_score': 0.7}
        """
        response = self.chatbot.get_response(query)
        
        # Apply filters if provided
        if filters:
            if 'category' in filters:
                response['all_matches'] = [
                    r for r in response['all_matches'] 
                    if filters['category'].lower() in r['category'].lower()
                ]
            
            if 'min_score' in filters:
                response['all_matches'] = [
                    r for r in response['all_matches'] 
                    if r['score'] >= filters['min_score']
                ]
            
            # Update best match
            if response['all_matches']:
                response['best_match'] = response['all_matches'][0]
            else:
                response['best_match'] = None
                response['response'] = "No results found matching your filters."
        
        return response


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "models" / "fine_tuned_legal_bert"
    data_path = base_dir / "data" / "final_merged_legal_qa.csv"
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please run train_model.py first to train the model.")
        exit(1)
    
    # Initialize chatbot
    chatbot = LegalChatbot(str(model_path), str(data_path))
    
    # Test queries
    test_queries = [
        "What if I stabbed someone with a knife?",
        "What are the essentials of a valid contract?",
        "What is Section 307 IPC?",
        "What is the punishment for attempt to murder?",
    ]
    
    print("\n" + "="*70)
    print("🧪 TESTING LEGAL CHATBOT")
    print("="*70)
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        response = chatbot.get_response(query)
        print(response['response'])
        print("\n" + "-"*70)
    
    # Start interactive session
    chatbot.interactive_chat()
