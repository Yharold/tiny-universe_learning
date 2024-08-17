import PyPDF2
import markdown
import json
import tiktoken
from bs4 import BeautifulSoup
import re
import os

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    def __init__(self, path: str) -> None:
        self.path = path
        self.file_list = self.get_files()

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

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        for file in self.file_list:
            # content：str
            content = self.read_file_content(file)
            # chunk_content:List[str]
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content
            )
            docs.extend(chunk_content)
        return docs

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

    @classmethod
    def read_pdf(cls, file_path: str):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

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

    @classmethod
    def read_text(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()


class Documents:
    def __init__(self, path: str = "") -> None:
        self.path = path

    def get_content(self):
        with open(self.path, mode="r", encoding="utf-8") as f:
            content = json.load(f)
        return content
