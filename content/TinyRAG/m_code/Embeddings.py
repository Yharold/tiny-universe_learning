from lib2to3.pytree import Base
import os
from typing import List
import numpy as np
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


class BaseEmbeddings:

    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class JinaEmbedding(BaseEmbeddings):
    def __init__(
        self, path: str = "jinaai/jina-embeddings-v2-base-zh", is_api: bool = False
    ) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model()

    def load_model(self):
        print("============Model Jina Loading============")
        import torch
        from transformers import AutoModel

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device)
        print("============M Jina Loaded============")
        return model

    def get_embedding(self, text: str) -> List[float]:
        return self._model.encode([text])[0].tolist()


class OpenAIEmbedding(BaseEmbeddings):
    def __init__(self, path: str = "", is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI  # type: ignore

            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embedding(
        self, text: str, model: str = "text-embedding-3-large"
    ) -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return (
                self.client.embeddings.create(input=[text], model=model)
                .data[0]
                .embedding
            )
        else:
            raise NotImplementedError
