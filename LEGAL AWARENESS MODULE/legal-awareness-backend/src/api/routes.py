from flask import Blueprint, request, jsonify
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot import LegalChatbot, LegalRetrievalSystem

api_bp = Blueprint('api', __name__)

# Initialize chatbot (lazy loading)
_chatbot = None
_retrieval_system = None

def get_chatbot():
    """Lazy load the chatbot"""
    global _chatbot
    if _chatbot is None:
        base_dir = Path(__file__).parent.parent.parent
        model_path = base_dir / "models" / "fine_tuned_legal_bert"
        data_path = base_dir / "data" / "final_merged_legal_qa.csv"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        _chatbot = LegalChatbot(str(model_path), str(data_path))
    return _chatbot

def get_retrieval_system():
    """Lazy load the retrieval system"""
    global _retrieval_system
    if _retrieval_system is None:
        base_dir = Path(__file__).parent.parent.parent
        model_path = base_dir / "models" / "fine_tuned_legal_bert"
        data_path = base_dir / "data" / "final_merged_legal_qa.csv"
        _retrieval_system = LegalRetrievalSystem(str(model_path), str(data_path))
    return _retrieval_system

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Legal Awareness Backend",
        "version": "1.0.0"
    }), 200

@api_bp.route('/ask', methods=['POST'])
def ask_question():
    """
    Ask a legal question
    Body: {"question": "What is Section 307 IPC?", "top_k": 3}
    """
    try:
        data = request.json
        question = data.get('question')
        top_k = data.get('top_k', 3)

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Get chatbot response
        chatbot = get_chatbot()
        response = chatbot.get_response(question, top_k=top_k)

        return jsonify({
            "success": True,
            "query": response['query'],
            "response": response['response'],
            "best_match": response['best_match'],
            "is_criminal_query": response['is_criminal_query'],
            "all_matches": response['all_matches']
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@api_bp.route('/search', methods=['POST'])
def search():
    """
    Advanced search with filters
    Body: {
        "query": "murder",
        "filters": {
            "category": "Criminal Law",
            "min_score": 0.7
        }
    }
    """
    try:
        data = request.json
        query = data.get('query')
        filters = data.get('filters', {})

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Get retrieval system
        retrieval = get_retrieval_system()
        response = retrieval.search(query, filters=filters)

        return jsonify({
            "success": True,
            "query": response['query'],
            "response": response['response'],
            "best_match": response['best_match'],
            "results": response['all_matches']
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@api_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get available legal categories"""
    try:
        chatbot = get_chatbot()
        categories = chatbot.qa_data['category'].unique().tolist()
        
        return jsonify({
            "success": True,
            "categories": categories,
            "total": len(categories)
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        chatbot = get_chatbot()
        
        stats = {
            "total_qa_pairs": len(chatbot.qa_data),
            "categories": chatbot.qa_data['category'].value_counts().to_dict(),
            "criminal_queries_count": chatbot.qa_data['category'].str.contains('Criminal', case=False).sum()
        }
        
        return jsonify({
            "success": True,
            "stats": stats
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500