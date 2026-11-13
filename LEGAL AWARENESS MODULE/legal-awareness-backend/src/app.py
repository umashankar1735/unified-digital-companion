from flask import Flask
from flask_cors import CORS
from api.routes import api_bp
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def create_app():
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    
    @app.route('/')
    def home():
        return {
            "service": "Legal Awareness Backend",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/api/v1/health",
                "ask": "/api/v1/ask",
                "search": "/api/v1/search",
                "categories": "/api/v1/categories",
                "stats": "/api/v1/stats"
            }
        }
    
    return app

if __name__ == "__main__":
    app = create_app()
    print("\n" + "="*70)
    print("⚖️  Legal Awareness Backend Server")
    print("="*70)
    print("Server running on: http://localhost:5000")
    print("API endpoints available at: http://localhost:5000/api/v1/")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)