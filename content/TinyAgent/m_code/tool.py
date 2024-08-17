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

    def baidu_search(self, search_query: str):
        pass
