# TinyAgent笔记

整个项目逻辑比较简单，但很有用。它确实解决了我对于Agent这个东西的一个疑惑。以前在网上看Agent的时候，概念都懂，但总是不太理解这个东西具体是个什么。有点像只在书里看过某个水果，但书里描述的再好，都不可能对这个水果的味道有一个正确的感受。

在我看来，所谓的Agent其实就是一个设计好的工具，AI是核心但不是全部，整个结构的设计同样重要。打个比方，AI就是一台发动机。我们每个人都知道发动机潜能巨大，但现在没人知道这个东西最擅长干什么。然后有大神就用发动机、轮胎、钢板，做了个东西出来，起了个名字，叫做“车”。人一看，这东西好，利用车我们可以日行千里。这个车就是Agent。

我们现在有了大模型，它的智能有点超出想象。但和发动机不同的是，发动机一开始就是冲着车去研发的，等发动机出来我们直接无缝衔接研发车。而AI这个东西现在没有找到它最合适的用处。每个人都知道AI潜力巨大但不知道把它用在哪里。一个基础的想法就是利用模型改进现有的工具。所以，Agent出来了。

我觉得Agent不会是什么“新东西”。因为AI目前在创新方面，除了alphafold之外好像还真没哪个是新的，是超出“老工具”的。比如SD，现在社会上确画画的人吗？比如ChatGPT，现在社会上缺聊天的工具吗？还是缺搜索的工具？顶多这些工具不好用，谈不上缺。甚至目前的AI还无法代替这些工具。

Agent是通过利用现有的一些工具，结合AI，创建一个更好的工具。这个项目就是通过InternLM2这个AI，结合谷歌搜索，得到更好的搜索结果。

整个模型分为三部分：

- 模型：包括模型的实现和使用方法。实现文件LLM.py
- 工具：工具的实现和使用方法。实现文件tool.py
- Agent: 统合模型和工具的文件，实现文件Agent.py,实现方法text_completion

## 模型

模型没什么说的。其中包括了加载模型的方法load_model和使用模型的方法chat

```python
from typing import List
from urllib import response
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

class BaseModel:
    def __init__(self,path:str="") -> None:
        self.path = path
        pass
    def chat(self,prompt:str,history:List[dict]):
        pass
    def load_model(self):
        pass
    
class InternLM2Chat(BaseModel):
    def __init__(self,path:str="") -> None:
        super().__init__(path)
        self.load_model()
        
    def load_model(self):
        print("================Loading model================")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path,torch_dtype=torch.float16,trust_remote_code=True).cuda().eval()
        print("================model loaded================")
    def chat(self,prompt:str,history:List[dict],meta_instruction:str="") ->str:
        response,history = self.model.chat(self.tokenizer,prompt,history,temperature=0.1,meta_instruction=meta_instruction)
        return response,history   
```

## 工具

工具也没什么说的，包括了工具的描述tools和对应的使用方法google_search

```python
import json, requests
class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()

    def _tools(self):
        tools = [
            {
                "name_for_human": "谷歌搜索",
                "name_for_model": "google_search",
                "description_for_model": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。",
                "parameters": [
                    {
                        "name": "search_query",
                        "description": "搜索关键词或短语",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
            },
            # {
            #     "name_for_human": "百度搜索",
            #     "name_for_model": "baidu_search",
            #     "description_for_model": "百度搜索是个垃圾",
            #     "parameters": [
            #         {
            #             "name": "search_query",
            #             "descripion": "搜索关键词",
            #             "required": True,
            #             "schema": {"type": "string"},
            #         }
            #     ],
            # },
        ]
        return tools

    def google_search(self, search_query: str):
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": search_query})
        headers = {
            "X-API-KEY": "5a1019288d55718a45a80ffaa1b3588c96c54427",
            "Content-Type": "application/json",
        }
        resopnse = requests.request("POST", url, headers=headers, data=payload).json()
        return resopnse["organic"][0]["snippet"]

    # def baidu_search(self, search_query: str):
    #     pass
```

## Agent

Agent的重点是模型和工具的设计模式。TinyAgent的设计模型如下：

- 规定模型输入prompt和输出的模板：实现方法build_system_input
- 设计了模型和工具的结构：输入——>模型——>工具——>模型——>输出
- 设计了工具的选择方法：实现方法parse_latest_plugin_call。这里工具的选择是由模型决定的。然后根据模型的输出选择工具。

整个TinyAgent的核心无疑是模型，因为任何工具的选择都取决于模型的输出。整个Agent的实现方法就是text_completion

```python
import json5
from m_code.LLM import InternLM2Chat
from m_code.tool import Tools

TOOL_DESC = """{name_for_model}:Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


class Agent:
    def __init__(self, path: str = "") -> None:
        self.path = path
        self.tool = Tools()
        self.system_prompt = self.build_system_input()
        self.model = InternLM2Chat(path)

    def build_system_input(self):
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool["name_for_model"])
        tool_descs = "\n\n".join(tool_descs)
        tool_names = ",".join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt

    def parse_latest_plugin_call(self, text):
        plugin_name, plugin_args = "", ""
        i = text.rfind("\nAction:")
        j = text.rfind("\nAction Input:")
        k = text.rfind("\nObservation:")
        if 0 <= i < j:
            if k < j:
                text = text.rstrip() + "\nObservation:"
            k = text.rfind("\nObservation:")
            plugin_name = text[i + len("\nAction:") : j].strip()
            plugin_args = text[j + len("\nAction Input:") : k].strip()
            text = text[:k]
            return plugin_name, plugin_args, text

    def call_plugin(self, plugin_name, plugin_args):
        plugin_args = json5.loads(plugin_args)
        if plugin_name == "google_search":
            return "\nObservation:" + self.tool.google_search(**plugin_args)

    def text_completion(self, text, history=[]):
        text = "\nQuestion:" + text
        response, his = self.model.chat(text, history, self.system_prompt)
        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)
        if plugin_name:
            response += self.call_plugin(plugin_name, plugin_args)
        response, his = self.model.chat(response, history, self.system_prompt)
        return response, his
```

## 总结

Agent的核心是模型，但结构的设计也至关重要。模型一定要根据对应的工具进行微调训练，不然不会有好的输出结构。