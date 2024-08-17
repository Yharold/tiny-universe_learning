from m_code.VectorBase import VectorStore
from m_code.utils import ReadFiles
from m_code.LLM import OpenAIChat, InternLMChat
from m_code.Embeddings import JinaEmbedding
import torch

# 建立向量数据库
docs = ReadFiles("./content/TinyRAG/m_data").get_content(
    max_token_len=600, cover_content=150
)  # 获得data目录下的所有文件内容并分割


vector = VectorStore(docs)
embedding = JinaEmbedding(
    path="jinaai/jina-embeddings-v2-base-zh"
)  # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(
    path="storage"
)  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

vector = VectorStore()

vector.load_vector("./storage")  # 加载本地的数据库

embedding = JinaEmbedding(path="jinaai/jina-embeddings-v2-base-zh")

question = "孙悟空在哪里打红孩儿？"

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]

if embedding._model.device.type == "cuda":
    embedding._model = None
    torch.cuda.empty_cache()

print(content)
path = "internlm/internlm2-chat-1_8b"
model = InternLMChat(path=path)
print(model.chat(question, [], content))
print("")
