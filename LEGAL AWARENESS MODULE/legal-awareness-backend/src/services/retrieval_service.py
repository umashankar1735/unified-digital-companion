from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

class RetrievalService:
    def __init__(self, model_path, dataset_path):
        self.model = SentenceTransformer(model_path)
        self.dataset = pd.read_csv(dataset_path)
        self.embeddings = self.model.encode(self.dataset["question"].tolist(), convert_to_tensor=True)

    def retrieve_answer(self, user_query, top_k=1, threshold=0.65):
        query_embedding = self.model.encode([user_query], convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_k_idx = torch.topk(cosine_scores, k=top_k).indices.tolist()

        best_idx = top_k_idx[0]
        best_score = float(cosine_scores[best_idx])

        if best_score < threshold:
            return None, best_score

        best_question = self.dataset.iloc[best_idx]["question"]
        best_answer = self.dataset.iloc[best_idx]["answer"]

        return best_answer, best_score