from sentence_transformers import SentenceTransformer
import openai
import os

class GPTService:
    def __init__(self):
        self.model_path = "/content/fine_tuned_inlegalbert"
        self.model = SentenceTransformer(self.model_path)
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_response(self, user_query, best_question, best_answer):
        prompt = f"""
        You are an Indian legal expert. Use the following dataset QA pair to help answer the user's question.

        Dataset Question: {best_question}
        Dataset Answer: {best_answer}

        User Query: {user_query}

        Based on Indian law (IPC, CrPC, and related acts), provide a detailed, well-reasoned answer in simple terms.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful Indian legal assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"⚠️ OpenAI API Error: {e}"