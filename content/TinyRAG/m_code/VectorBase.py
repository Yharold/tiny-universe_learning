import os, json
from typing import List
from m_code.Embeddings import BaseEmbeddings
import numpy as np
from tqdm import tqdm


class VectorStore:
    def __init__(self, document: List[str] = [""]):
        self.document = document

    # 将文档document中的所有字符串转为对应的向量
    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    # 将文档document和向量vectors保存为json文件
    def persist(self, path: str = "storage"):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/document.json", "w", encoding="utf-8") as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", "w", encoding="utf-8") as f:
                json.dump(self.vectors, f)

    # 载入文档和向量
    def load_vector(self, path: str = "storage"):
        with open(f"{path}/vectors.json", "r", encoding="utf-8") as f:
            self.vectors = json.load(f)
        with open(f"{path}/document.json", "r", encoding="utf-8") as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    # 查找与问题query相似度最高的k个内容，这些内容保存在document中
    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        # 计算相似度
        result = np.array(
            [self.get_similarity(query_vector, vector) for vector in self.vectors]
        )
        # result.argsort() 对相似度排序，从低到高
        # result.argsort()[-k:] 拿到最后k个索引
        # result.argsort()[-k:][::-1] 倒序一下，从高到低
        # np.array(self.document) 将self.document转为np数组
        # np.array(self.document)[result.argsort()[-k:][::-1]] 拿到了这个数组中的对应的内容
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
