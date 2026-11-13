from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)