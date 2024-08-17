# TinyRAG笔记

整个项目的逻辑如下：
1. 文件处理：包括读取与分段。对一个文件进行读取，读取后再进行分段。
2. 编码数据：对分段后的数据进行编码，这个也是通过模型进行处理的。
3. 数据存储：对源数据和编码后的数据进行存储，要实现读取，存储和查找三个功能。
4. 搜索模型：这个是核心模型，也就是通过编码数据和问题，回答问题的模型。功能主要包括：载入模型，问答。

所以整个RAG其实就是使用搜索模型进行问答。

## 文件处理

文件处理包括读取与分段。读取就不说了，说下分段。分段需要根据预设的最大长度进行分段，每段之间还需要重复一部分。另外每段可以短一些，但不要在半中间分段。比如`整天和山中的动物一起玩乐，过得十分快活。一天，天气特别热，猴子们为了躲避炎热的天气，跑到山涧里洗澡。它们看见这泉水哗哗地流，就顺着涧往前走，去寻找它的源头。猴子们爬呀、爬呀，走到了尽头，却看见一股瀑布，像是从天而降一样。`，如果按照最大长度分段，可能这样的`整天和山中的动物一起玩乐，过得十分快活。一天，天气特别热，猴子们为了躲避炎热的天气，跑到山涧里洗澡。它们看见这泉水哗哗地流，就顺着涧往前走，去寻找它的源头。猴子们爬呀、爬呀，走到了尽头，却看见一`。最后一句话没说完。其实分成这样更好`整天和山中的动物一起玩乐，过得十分快活。一天，天气特别热，猴子们为了躲避炎热的天气，跑到山涧里洗澡。它们看见这泉水哗哗地流，就顺着涧往前走，去寻找它的源头。`。也就是可以多分几段，每段短一些，不要有断句。

```python
import PyPDF2
import markdown
import json
import tiktoken
from bs4 import BeautifulSoup
import re
import os

enc = tiktoken.get_encoding("cl100k_base")

# 读取文件
class ReadFiles:
    def __init__(self, path: str) -> None:
        self.path = path
        self.file_list = self.get_files()
    # 搜索路径下的所有可处理文件，并返回文件的路径
    def get_files(self):
        file_list = []
        for (
            filepath,
            dirnames,
            filenames,
        ) in os.walk(self.path):
            for filename in filenames:
                if filename.endswith(".md"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list
    # 读取文件并对文件进行分段
    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        for file in self.file_list:
            # content：str 读取文件
            content = self.read_file_content(file)
            # chunk_content:List[str]
            #对文件进行分段
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content
            )
            # 分段后的数据全部放进docs中
            docs.extend(chunk_content)
        return docs
    # 对文件进行分段
    # max_token_len:每段的最大长度
    # cover_content:每段之间重复的token数
    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []
        curr_len = 0
        curr_chunk = ""
        token_len = max_token_len - cover_content
        lines = text.splitlines()
        for line in lines:
            line = line.replace(" ", "")
            line_len = len(enc.encode(line))
            if line_len > max_token_len:
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    # 避免跨单词分割
                    while not line[start:end].rstrip().isspace():
                        start += 1
                        end += 1
                        if start >= line_len:
                            break
                    # 上一个后面cover_content+当前的line
                    curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                    chunk_text.append(curr_chunk)
            if curr_len + line_len <= token_len:
                curr_chunk += line
                curr_chunk += "\n"
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line
                curr_len = line_len + cover_content
        if curr_chunk:
            chunk_text.append(curr_chunk)
        return chunk_text
    # 读取文件
    @classmethod
    def read_file_content(cls, file_path: str):
        if file_path.endswith(".pdf"):
            return cls.read_pdf(file_path)
        elif file_path.endswith(".md"):
            return cls.read_markdown(file_path)
        elif file_path.endswith(".txt"):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")
    # 读取pdf
    @classmethod
    def read_pdf(cls, file_path: str):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text
    # 读取md
    @classmethod
    def read_markdown(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # 使用bs提取文本
            soup = BeautifulSoup(html_text, "html.parser")
            plain_text = soup.get_text()
            text = re.sub(r"http\S+", "", plain_text)
            return text
    # 读取txt
    @classmethod
    def read_text(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

# 设计时是用来读取json文档，但实际中没用
class Documents:
    def __init__(self, path: str = "") -> None:
        self.path = path

    def get_content(self):
        with open(self.path, mode="r", encoding="utf-8") as f:
            content = json.load(f)
        return content

```

## 编码数据

编码数据使用的是AI模型，不管是本地模型还是使用api都可以。功能无非就是**载入模型**或者**加载api**和**使用模型**。

```python
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

```

## 数据存储

其实这个事应该是使用数据库，但这里为了简单直接使用了文件进行保存。

主要需要实现的功能有：
- 利用编码模型进行编码
- 保存源数据和编码数据
- 读取源数据和编码数据
- 计算两个编码数据的相似度
- 根据问题查找最相关的源数据

这里面主要讲后面两个。计算两个编码数据的相似度使用的就是点积，这个功能是在编码模型中实现的，这里只是调用。根据问题查找最相关的数据方法就是先将问题转为编码数据，然后在对应文档中计算相似度，最后根据相似度给出源数据

```python
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
```

## 搜索模型

这个模型是整个关键，但实现起来却挺简单。因为难点在于模型的训练。代码主要实现的就两个**模型加载**和**模型使用**

```python
import os
from typing import Dict, List

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
)


class BaseModel:
    def __init__(self, path: str = "") -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class OpenAIChat(BaseModel):
    def __init__(self, path: str = "", model: str = "gpt-3.5-turbo-1106") -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        client.base_url = os.getenv("OPENAI_BASE_URL")
        history.append(
            {
                "role": "user",
                "content": PROMPT_TEMPLATE["RAG_PROMPT_TEMPALTE"].format(
                    question=prompt, context=content
                ),
            }
        )
        response = client.chat.completions.create(
            model=self.model, messages=history, max_tokens=150, temperature=0.1
        )
        return response.choices[0].message.content


class InternLMChat(BaseModel):
    def __init__(self, path: str = "") -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str = "") -> str:
        prompt = PROMPT_TEMPLATE["InternLM_PROMPT_TEMPALTE"].format(
            question=prompt, context=content
        )
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response

    def load_model(self):
        print("============Model InternLM Loading============")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)
        print("============Model InternLM Loaded============")
```

## 总结

整个模型还是比较简洁的。但同样有些问题没有细说。比如根据问题搜索最相关的数据，项目中是直接在整个保存的文件中搜索，但实际中不可能，因为源数据太大了，如何缩小搜索范围同样是一个重点。我猜测是对整个文档进行一个编码，然后使用模型进行搜索，就像我们根据书名缩小找书的范围。

