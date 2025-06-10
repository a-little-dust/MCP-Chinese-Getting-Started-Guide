import httpx
from mcp.server import FastMCP #这在github开源了，用于构建mcp服务器

# # 初始化 FastMCP 服务器
app = FastMCP('web-search')

@app.tool() #把函数注册为mcp工具 
async def web_search(query: str) -> str:
    """
    搜索互联网内容

    Args:
        query: 要搜索内容

    Returns:
        搜索结果的总结
    """

    async with httpx.AsyncClient() as client:
        response = await client.post(
            'https://open.bigmodel.cn/api/paas/v4/tools',
            headers={'Authorization': '换成你自己的API KEY'},
            json={
                'tool': 'web-search-pro',
                'messages': [
                    {'role': 'user', 'content': query}
                ],
                'stream': False
            }#根据API文档，构建请求
        )

        res_data = []
        for choice in response.json()['choices']:
            for message in choice['message']['tool_calls']:
                search_results = message.get('search_result')
                if not search_results:
                    continue
                for result in search_results:
                    res_data.append(result['content'])

        return '\n\n\n'.join(res_data)

@app.tool()
async def hello_world(name: str) -> str:
    """
    打招呼
    """
    return f"Hello, {name}!"



if __name__ == "__main__":
    app.run(transport='stdio')