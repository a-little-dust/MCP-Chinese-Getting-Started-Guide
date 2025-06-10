import json
import asyncio
import os
from typing import Optional
from contextlib import AsyncExitStack

from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


load_dotenv()


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None #用于存储与服务器的会话
        self.exit_stack = AsyncExitStack()#用于管理资源的清理
        self.client = OpenAI()#是 OpenAI 客户端的实例，用于与 OpenAI API 进行交互
    async def connect_to_server(self):#连接到mcp服务器
        server_params = StdioServerParameters(
            command='uv',
            args=['run', 'web_search.py'],
            env=None
        )#创建一个服务器参数对象，用于配置mcp服务器，运行web_search.py
        # 这个文件里可以有多个工具

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))#基于server_params创建一个stdio_client对象
            # 通过exit_stack管理资源，确保在退出时正确关闭
        stdio, write = stdio_transport#解包传输对象，获取标准输入和输出
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write))#基于stdio和write创建一个ClientSession对象

        await self.session.initialize()#初始化会话
    async def process_query(self, query: str) -> str:
        # 这里需要通过 system prompt 来约束一下大语言模型，
        # 否则会出现不调用工具，自己乱回答的情况
        system_prompt = (
            "You are a helpful assistant."
            "You have the function of online search. "
            "Please MUST call web_search tool to search the Internet content before answering."
            "Please do not lose the user's question information when searching,"
            "and try to maintain the completeness of the question content as much as possible."
            "When there is a date related question in the user's question," #如果用户的问题中包含日期相关信息，
            "please use the search function directly to search and PROHIBIT inserting specific time."#请直接使用搜索功能进行搜索，并禁止插入具体时间
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # 获取所有 mcp 服务器 工具列表信息
        response = await self.session.list_tools()
        # 根据list_tools，生成 function call 的描述信息
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response.tools]

        # 请求 deepseek，function call 的描述信息通过 tools 参数传入
        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=messages,
            tools=available_tools
        )

        # 处理返回的内容
        content = response.choices[0]
        if content.finish_reason == "tool_calls":#如果需要使用工具，就解析工具
            # 如何是需要使用工具，就解析工具
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # 执行工具
            result = await self.session.call_tool(tool_name, tool_args)#通过session.call_tool调用工具
            print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")
			
            # 将 deepseek 返回的调用哪个工具数据和工具执行完成后的数据都存入messages中
            messages.append(content.message.model_dump())#把这个关于工具调用的Message放入messages中
            messages.append({
                "role": "tool",
                "content": result.content[0].text,#工具调用的结果
                "tool_call_id": tool_call.id,
            })

            # 将上面的结果再返回给 deepseek 用于生产最终的结果（deepseek是一个大语言模型）
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL"),
                messages=messages,
            )
            return response.choices[0].message.content#返回最终的结果

        return content.message.content#如果不需要使用工具，就返回最终的结果
    async def chat_loop(self):
        while True:
            try:
                query = input("\nQuery: ").strip()#获取用户输入

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)#调用包含MCP的模型
                print("\n" + response)

            except Exception as e:
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()#关闭所有资源
    async def main():
        client = MCPClient()#创建一个MCPClient对象
        try:
            await client.connect_to_server()#连接到mcp服务器
            await client.chat_loop()#开始聊天循环
        finally:
            await client.cleanup()#清理资源


if __name__ == "__main__":
    import sys

    asyncio.run(main())
