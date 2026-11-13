from typing import List
import pandas as pd

class QADataset:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_data(self) -> None:
        self.data = pd.read_csv(self.file_path)

    def get_questions(self) -> List[str]:
        if self.data is not None:
            return self.data['question'].tolist()
        return []

    def get_answers(self) -> List[str]:
        if self.data is not None:
            return self.data['answer'].tolist()
        return []

    def get_data(self) -> pd.DataFrame:
        return self.data if self.data is not None else pd.DataFrame()