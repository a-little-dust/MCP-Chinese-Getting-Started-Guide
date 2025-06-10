# Model Context Protocol(MCP) 编程极速入门

## 简介

模型上下文协议（MCP）是一个创新的开源协议，它重新定义了大语言模型（LLM）与外部世界的互动方式。MCP 提供了一种标准化方法，使任意大语言模型能够轻松连接各种数据源和工具，实现信息的无缝访问和处理。MCP 就像是 AI 应用程序的 USB-C 接口，为 AI 模型提供了一种标准化的方式来连接不同的数据源和工具。

![image-20250223214308430](.assets/image-20250223214308430.png)

通过本文，学到了：创建MCP服务端；创建客户端，连接MCP；给cursor等连接上MCP

MCP 有以下几个核心功能：

- Resources 资源
- Prompts 提示词
- Tools 工具
- Sampling 采样
- Roots 根目录
- Transports 传输层

因为大部分功能其实都是服务于 Claude 客户端的，本文更希望编写的 MCP 服务器服务与通用大语言模型，所以本文将会主要以“工具”为重点，其他功能会放到最后进行简单讲解。

其中 MCP 的传输层支持了 2 种协议的实现：stdio（标准输入/输出）和 SSE（服务器发送事件），因为 stdio 更为常用，所以本文会以 stdio 为例进行讲解。

本文将会使用 3.11 的 Python 版本，并使用 uv 来管理 Python 项目。同时代码将会在文末放到 Github 上，废话不多说，我们这就开始吧~

## 开发 MCP 服务器

在这一小节中，我们将会实现一个用于网络搜索的服务器。首先，我们先来通过 uv 初始化我们的项目。

> uv 官方文档：https://docs.astral.sh/uv/

```shell
# 初始化项目
uv init mcp_getting_started
cd mcp_getting_started

# 创建虚拟环境并进入虚拟环境
uv venv
.venv\Scripts\activate.bat

# 安装依赖
uv add "mcp[cli]" httpx openai

```

然后我们来创建一个叫 `web_search.py` 文件，来实现我们的服务。MCP 为我们提供了2个对象：`mcp.server.FastMCP` 和 `mcp.server.Server`，`mcp.server.FastMCP` 是更高层的封装，我们这里就来使用它。


## 调试 MCP 服务器

此时，我们就完成了 MCP 服务端的编写。下面，我们来使用官方提供的 `Inspector` 可视化工具来调试我们的服务器。

我们可以通过两种方法来运行`Inspector`：

> 请先确保已经安装了 node 环境。

通过 npx：

```shell
npx -y @modelcontextprotocol/inspector <command> <arg1> <arg2>
```

我们的这个代码运行命令为：

```shell
npx -y @modelcontextprotocol/inspector uv run web_search.py
```



通过 mcp dev 来运行：

```shell
mcp dev PYTHONFILE
```

我们的这个代码运行命令为：

```shell
mcp dev web_search.py
```

当出现如下提示则代表运行成功。如果提示连接出错，可能是端口被占用，可以看这个 issue 的解决方法：https://github.com/liaokongVFX/MCP-Chinese-Getting-Started-Guide/issues/6

![image-20250223223638135](.assets/image-20250223223638135.png)

然后，我们打开这个地址，点击左侧的 `Connect` 按钮，即可连接我们刚写的服务。然后我们切换到 `Tools` 栏中，点击 `List Tools` 按钮即可看到我们刚写的工具，我们就可以开始进行调试啦。

![image-20250223224052795](.assets/image-20250223224052795.png)


## Sampling 讲解

MCP 还为我们提供了一个 `Sampling` 的功能，这个如果从字面来理解会让人摸不到头脑，但实际上这个功能就给了我们一个在执行工具的前后的接口，我们可以在工具执行前后来执行一些操作。比如，当调用本地文件的删除的工具的时候，肯定是期望我们确认后再进行删除。那么，此时就可以使用这个功能。

下面我们就来实现这个人工监督的小功能。

首先，我们来创建个模拟拥有删除文件的 MCP 服务器：

```python
# 服务端
from mcp.server import FastMCP
from mcp.types import SamplingMessage, TextContent

app = FastMCP('file_server')


@app.tool()
async def delete_file(file_path: str):
    # 创建 SamplingMessage 用于触发 sampling callback 函数
    # 获取当前会话，然后发送消息SamplingMessage，内容是 询问是否删除文件
    result = await app.get_context().session.create_message(
        messages=[
            SamplingMessage(
                role='user', content=TextContent(
                    type='text', text=f'是否要删除文件: {file_path} (Y)')
            )
        ],
        max_tokens=100
    )

    # 获取到 sampling callback 函数的返回值，并根据返回值进行处理
    if result.content.text == 'Y':
        return f'文件 {file_path} 已被删除！！'


if __name__ == '__main__':
    app.run(transport='stdio')

```

这里最重要的就是需要通过`create_message`方法来创建一个 `SamplingMessage` 类型的 message，他会将这个 message 发送给 sampling callback 对应的函数中。
每个消息类型都有对应的回调函数，函数的内容由客户端来定义

接着，我们来创建客户端的代码：

```python
# 客户端
import asyncio

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from mcp.shared.context import RequestContext
from mcp.types import (
    TextContent,
    CreateMessageRequestParams,
    CreateMessageResult,
)

server_params = StdioServerParameters(
    command='uv',
    args=['run', 'file_server.py'],
)#这里可以看出 这个文件运行的是客户端


async def sampling_callback(# 定义回调函数
        context: RequestContext[ClientSession, None],
        params: CreateMessageRequestParams,
):
    # 获取工具发送的消息并显示给用户，通过input()接受用户的输入
    input_message = input(params.messages[0].content.text)
    # 将用户输入发送回工具
    return CreateMessageResult(
        role='user',
        content=TextContent(
            type='text',
            text=input_message.strip().upper() or 'Y'
        ),
        model='user-input',
        stopReason='endTurn'
    )


async def main():
    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(#除了前面设置的标准输入、标准输出，还要给client设置回调函数
                stdio, write,
                # 设置 sampling_callback 对应的方法
                sampling_callback=sampling_callback
        ) as session:
            await session.initialize()
            res = await session.call_tool(
                'delete_file',
                {'file_path': 'C:/xxx.txt'}
            )
            # 获取工具最后执行完的返回结果
            print(res)


if __name__ == '__main__':
    asyncio.run(main())

```

特别要注意的是，目前在工具里面打印的内容实际上使用 `stdio_client` 是无法显示到命令行窗口的。所以，我们调试的话，可以使用 `mcp.shared.memory.create_connected_server_and_client_session`来替代 stdio_client，这个函数可以创建一个内存中的连接，使得调试信息能够正常显示

具体代码如下：

```python
# 客户端
from mcp.shared.memory import (
    create_connected_server_and_client_session as create_session
)
# 这里需要引入服务端的 app 对象
from file_server import app

async def sampling_callback(context, params):
    ...

async def main():
    # 使用 create_connected_server_and_client_session 创建连接
    async with create_session(
        app._mcp_server,
        sampling_callback=sampling_callback
    ) as client_session:
        await client_session.call_tool(
            'delete_file', 
            {'file_path': 'C:/xxx.txt'}
        )

if __name__ == '__main__':
    asyncio.run(main())
```



## Claude Desktop 加载 MCP Server

Claude 桌面端是一个功能强大的 AI 助手应用程序，可以与 Claude AI 进行自然语言对话，可以加载自定义的 MCP 服务器

因为后面的两个功能实际上都是为了提供给 Claude 桌面端用的，所以这里先说下如何加载我们自定义的 MCP Server 到 Claude 桌面端。

首先，我们先打开配置。

![image-20250227221154638](.assets/image-20250227221154638.png)

我们点击 `Developer` 菜单，然后点击 `Edit Config` 按钮打开 Claude 桌面端的配置文件 `claude_desktop_config.json`

![image-20250227221302174](.assets/image-20250227221302174.png)

然后开始添加我们的服务器，服务器需要在 `mcpServers` 层级下，参数有 `command`、`args`、`env`。实际上，参数和 `StdioServerParameters` 对象初始化时候的参数是一样的。 

```json
{
  "mcpServers": {
    "web-search-server": {
      "command": "uv",
      "args": [
        "--directory",
        "D:/projects/mcp_getting_started",
        "run",
        "web_search.py"
      ]
    }
  }
}
```

最后，我们保存文件后重启 Claude 桌面端就可以在这里看到我们的插件了。

![image-20250227221911231](.assets/image-20250227221911231.png)

![image-20250227221921036](.assets/image-20250227221921036.png)

当然，我们也可以直接在我们插件的目录下运行以下命令来直接安装：

```shell
mcp install web_search.py
```



## 其他功能

### Prompt

MCP 还为我们提供了一个生成 Prompt 模板的功能。他使用起来也很简单，只需要使用 `prompt` 装饰器装饰一下即可，代码如下：

```python
from mcp.server import FastMCP

app = FastMCP('prompt_and_resources')

@app.prompt('翻译专家')
async def translate_expert(
        target_language: str = 'Chinese',
) -> str:#定义提示词模板，返回str
    return f'你是一个翻译专家，擅长将任何语言翻译成{target_language}。请翻译以下内容：'


if __name__ == '__main__':
    app.run(transport='stdio')

```

然后我们用上一节讲到的配置 Claude 桌面端 MCP 服务器的方法添加下我们的新 MCP 服务器。然后我们就可以点击右下角的图标开始使用啦。

他会让我们设置一下我们传入的参数，然后会把提示词 根据模板 得到txt文件，后续会根据txt来回答

![mcp001](.assets/mcp001-1740666812436-2.gif)



### Resource

我们还可以在 Claude 客户端上选择我们为用户提供的预设资源，同时也支持自定义的协议。具体代码如下：

```python
from mcp.server import FastMCP

app = FastMCP('prompt_and_resources')

@app.resource('echo://static')
async def echo_resource():
    # 返回的是，当用户使用这个资源时，资源的内容
    return 'Echo!'

# 通配符资源 - 根据参数返回不同的内容，有别于静态资源
@app.resource('greeting://{name}')
async def get_greeting(name):
    return f'Hello, {name}!'


if __name__ == '__main__':
    app.run(transport='stdio')

```

然后，我们到 Claude 桌面端上看看。

![mcp002](.assets/mcp002.gif)

这里要特别注意的是，目前 Claude 桌面端是没法读到资源装饰器设置 `greeting://{name}` 这种通配符的路径，未来将会被支持。但是，在我们的客户端代码中是可以当做资源模板来使用的，具体代码如下：

```python
import asyncio
from pydantic import AnyUrl

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

server_params = StdioServerParameters(
    command='uv',
    args=['run', 'prompt_and_resources.py'],
)


async def main():
    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()

            # 获取无通配符的资源列表
            res = await session.list_resources()
            print(res)

            # 获取有通配符的资源列表(资源模板)
            res = await session.list_resource_templates()
            print(res)

            # 读取资源，会匹配通配符
            res = await session.read_resource(AnyUrl('greeting://liming'))
            print(res)

            # 获取 Prompt 模板列表
            res = await session.list_prompts()
            print(res)

            # 使用 Prompt 模板
            res = await session.get_prompt(
                '翻译专家', arguments={'target_language': '英语'})
            print(res)


if __name__ == '__main__':
    asyncio.run(main())

```



### 生命周期

MCP 生命周期分为3个阶段：

- 初始化
- 交互通信中
- 服务被关闭

因此，我们可以在这个三个阶段的开始和结束来做一些事情，比如创建数据库连接和关闭数据库连接、记录日志、记录工具使用信息等。

下面我们将以网页搜索工具，把工具调用时的查询和查询到的结果存储到一个全局上下文中作为缓存为例，来看看生命周期如何使用。完整代码如下：

```python
import httpx
from dataclasses import dataclass
from contextlib import asynccontextmanager

from mcp.server import FastMCP
from mcp.server.fastmcp import Context


@dataclass
# 初始化一个生命周期上下文对象
class AppContext:
    # 里面有一个字段用于存储请求历史，是字典类型
    histories: dict


@asynccontextmanager
async def app_lifespan(server):
    # 在 MCP 初始化时执行
    histories = {}
    try:
        # 每次通信会把这个上下文通过参数传入工具
        yield AppContext(histories=histories)
    finally:
        # 当 MCP 服务关闭时执行，打印历史消息
        print(histories)


app = FastMCP(
    'web-search', 
    # 设置生命周期监听函数
    lifespan=app_lifespan
)


@app.tool()
# 第一个参数会被传入上下文对象
async def web_search(ctx: Context, query: str) -> str:
    """
    搜索互联网内容

    Args:
        query: 要搜索内容

    Returns:
        搜索结果的总结
    """
    # 如果之前问过同样的问题、有这个key，就直接返回缓存
    histories = ctx.request_context.lifespan_context.histories
    if query in histories：
    	return histories[query]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            'https://open.bigmodel.cn/api/paas/v4/tools',
            headers={'Authorization': 'YOUR API KEY'},
            json={
                'tool': 'web-search-pro',
                'messages': [
                    {'role': 'user', 'content': query}
                ],
                'stream': False
            }
        )

        res_data = []
        for choice in response.json()['choices']:
            for message in choice['message']['tool_calls']:
                search_results = message.get('search_result')
                if not search_results:
                    continue
                for result in search_results:
                    res_data.append(result['content'])

        return_data = '\n\n\n'.join(res_data)

        # 将查询值和返回值存入到 histories 中
        ctx.request_context.lifespan_context.histories[query] = return_data
        return return_data


if __name__ == "__main__":
    app.run()

```



## 在 LangChain 中使用 MCP 服务器

最近 LangChain 发布了一个新的开源项目 `langchain-mcp-adapters`，可以很方便的将 MCP 服务器集成到 LangChain 中。下面我们来看看如何使用它:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")

server_params = StdioServerParameters(#定义 启动mcp server的参数
    command='uv',
    args=['run', 'web_search.py'],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # 基于MCP，获取工具列表（相关函数由langchain提供）
        tools = await load_mcp_tools(session)

        # 创建并使用 ReAct agent，传入tools
        agent = create_react_agent(model, tools)
        agent_response = await agent.ainvoke({'messages': '杭州今天天气怎么样？'})
```

更详细的使用方法请参考：https://github.com/langchain-ai/langchain-mcp-adapters



## DeepSeek + cline + 自定义MCP = 图文大师

最后，我们来使用 VsCode 的 cline 插件，来通过 DeepSeek 和我们自定义的一个图片生成的 mcp 服务器来构建一个图文大师的应用。废话不多说，我们直接开始。

Cline 是一个 VSCode 的 AI 编程助手插件，它集成了多个大语言模型，包括 DeepSeek 等，支持配置自定义的 MCP 服务器

首先先来构建我们的图片生成的 mcp server，这里我们直接用 huggingface 上的 `FLUX.1-schnell` 模型，地址是：https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell 。这里我们不使用 `gradio_client` 库，而是会使用 `httpx` 手搓一个，因为使用 `gradio_client` 库可能会出现编码错误的bug。具体代码如下：

```python
# image_server.py

import json
import httpx
from mcp.server import FastMCP


app = FastMCP('image_server')


@app.tool()
async def image_generation(image_prompt: str):
    """
    生成图片
    :param image_prompt: 图片描述，需要是英文
    :return: 图片保存到的本地路径
    """
    async with httpx.AsyncClient() as client:
        data = {'data': [image_prompt, 0, True, 512, 512, 3]}

        # 创建生成图片任务
        response1 = await client.post(
            'https://black-forest-labs-flux-1-schnell.hf.space/call/infer',
            json=data,
            headers={"Content-Type": "application/json"}
        )

        # 解析响应获取事件 ID
        response_data = response1.json()
        event_id = response_data.get('event_id')

        if not event_id:
            return '无法获取事件 ID'

        # 通过流式的方式拿到返回数据
        url = f'https://black-forest-labs-flux-1-schnell.hf.space/call/infer/{event_id}'
        full_response = ''
        async with client.stream('GET', url) as response2:
            async for chunk in response2.aiter_text():
                full_response += chunk

        return json.loads(full_response.split('data: ')[-1])[0]['url']

if __name__ == '__main__':
    app.run(transport='stdio')

```

然后我们可以在虚拟环境下使用下面的命令打开 MCP Inspector 进行调试下我们的工具。

```shell
mcp dev image_server.py
```

![image-20250301231332749](.assets/image-20250301231332749.png)

接着我们在 VsCode 中安装 cline 插件，当安装完插件后，我们配置一下我们的 deepseek 的 api key。接着，我们点击右上角的 `MCP Server` 按钮打开 mcp server 列表。

![image-20250301232248034](.assets/image-20250301232248034.png)

然后切换到 `Installed` Tab 点击 `Configure MCP Servers` 按钮来编辑自定义的 mcp 服务器。

![image-20250301232417966](.assets/image-20250301232417966.png)

配置如下：

```json
{
  "mcpServers": {
    "image_server": {
      "command": "uv",
      "args": [
        "--directory",
        "D:/projects/mcp_getting_started",
        "run",
        "image_server.py"
      ],
      "env": {},
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

我们保存后，这里的这个小点是绿色的就表示我们的服务器已连接，然后我们就可以开始使用啦。

![image-20250301232809433](.assets/image-20250301232809433.png)

然后，我们就打开输入框，来输入我们的要写的文章的内容：

![image-20250301233421292](.assets/image-20250301233421292.png)

我们可以看到，他正确的调用了我们的工具

![image-20250301233726301](.assets/image-20250301233726301.png)

最后，就是可以看到生成的文章啦。

![image-20250301234532249](.assets/image-20250301234532249.png)



## 借助 serverless 将 MCP 服务部署到云端

上面我们讲的都是如何使用本地的 MCP 服务，但是有时我们希望直接把 MCP 服务部署到云端来直接调用，就省去了本地下载启动的烦恼了。此时，我们就需要来使用 MCP 的 SSE 的协议来实现了。

SSE 是一种服务器推送技术，允许服务器向客户端推送数据
它是基于 HTTP 协议的单向通信机制
服务器可以持续向客户端发送数据流，而客户端只需要保持连接即可
优势：支持跨网络通信，可以把服务端部署到云服务器上，客户端不需要本地运行服务器

此时，我们先来写 SSE 协议的 MCP 服务。实现起来很简单，只需要将我们最后的 `run` 命令中的 `transport` 参数设置为 `sse` 即可。下面还是以上面的网络搜索为例子，来实现一下 ，具体代码如下：

```python
# sse_web_search.py
import httpx

from mcp.server import FastMCP


app = FastMCP('web-search', port=9000)


@app.tool()
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
            headers={'Authorization': 'YOUR API KEY'},
            json={
                'tool': 'web-search-pro',
                'messages': [
                    {'role': 'user', 'content': query}
                ],
                'stream': False
            }
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


if __name__ == "__main__":
    app.run(transport='sse')#核心改动

```

在 `FastMCP` 中，有几个可以设置 SSE 协议相关的参数：

- host: 服务地址，默认为 `0.0.0.0`
- port: 服务端口，默认为 8000。上述代码中，我设置为 `9000`
- sse_path：sse 的路由，默认为 `/sse`

此时，我们就可以直接写一个客户端的代码来进行测试了。具体代码如下：

```python
import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession


async def main():
    # 连接到服务器，用with确保正确关闭
    async with sse_client('http://localhost:9000/sse') as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()

            res = await session.call_tool('web_search', {'query': '杭州今天天气'})
            print(res)


if __name__ == '__main__':
    asyncio.run(main())

```

我们可以看到，他正常工作了，并搜索到了内容：

![image-20250406152518223](.assets/image-20250406152518223.png)

当然，我们也可以使用 `mcp dev sse_web_search.py` 的方式来测试。这里要注意的是，`Transport Type` 需要改成 `SSE`，然后下面填写我们的本地服务地址。

![image-20250406153106098](.assets/image-20250406153106098.png)

下面的内容是关于 如何把MCP服务放到阿里云服务器，让我们能通过服务器地址来访问这个MCP服务，而不是访问本地的服务。这样其他人也可以使用

当一切都测试没有问题后，我们就来将他通过 severless 的方式来部署到云端。这里我们选择的是阿里云的函数计算服务。首先我们先进入到阿里云的 `函数计算 FC 3.0` 的 `函数` 菜单，并点击 `创建函数` 来创建我们的服务。地址是：https://fcnext.console.aliyun.com/cn-hangzhou/functions

![image-20250406153655185](.assets/image-20250406153655185.png)

我们这里选择 `Web函数` ，运行环境我们选择 `Python 10`。代码上传方式这里可以根据大家需求来，因为我这里就一个 python 文件，所以我这里就直接选择`使用示例代码`了，这样我后面直接把我的代码覆盖进去了就行了。启动命令和监听端口我这里都保留为默认(**端口需要和代码中一致**)。

环境变量大家可以将代码中用到的 apikey 可以设置为一个环境变量，这里我就不设置了。最后设置完成截图如下：

![image-20250406154115438](.assets/image-20250406154115438.png)

在高级设置中，为了方便调试，我启动了日志功能。

![image-20250406154228341](.assets/image-20250406154228341.png)

设置完成后，点创建即可。他就跳转到代码编辑部分，然后我们把之前的代码复制进去即可。

![image-20250406154441634](.assets/image-20250406154441634.png)

完成后，我们来安装下依赖。我们点击右上角的`编辑层`。这里默认会有个默认的 flask 的层，因为开始的模板用的是 flask，这里我们就不需要了。我们删除他，再添加一个 mcp 的层。选择`添加官方公共层`，然后搜索 `mcp` 就能看到了一个 python 版的 MCP 层，里面包含了 MCP 所有用到的依赖。

![image-20250406154753623](.assets/image-20250406154753623.png)

如果你还有其他第三方的，可以先搜索下看看公共层中是否有，没有就可以自行构建一个自定义的层。点击这里就可以，只需要提供一个 `requirements` 列表就可以了，这里就不赘述了。 

![image-20250406154935751](.assets/image-20250406154935751.png)

当我们都设置完成后，点击右下角的部署即可。

然后我们又回到了我们代码编辑的页面，此时，我们再点击左上角的部署代码。稍等一两秒就会提示代码部署成功。此时，我们的 MCP 服务就被部署到了云端。

![image-20250406155135563](.assets/image-20250406155135563.png)



> 20250409 更新：不知道是不是官方看到了这篇文章，现在运行时可以直接选择 `MCP 运行时` 了，就不用再在层那里手动添加 `MCP 层` 了。
>
> ![image-20250409213302652](.assets/image-20250409213302652.png)



然后，我们切换到`配置`的`触发器`中，就可以看到我们用来访问的 URL 地址了。当然，你也可以绑定自己的域名。

![image-20250406155353662](.assets/image-20250406155353662.png)

然后，我们就可以用我们上面的客户端代码进行测试了。

```python
import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession


async def main():
    # 注意 这里只是把地址改变了
    async with sse_client('https://mcp-test-whhergsbso.cn-hangzhou.fcapp.run/sse') as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()

            res = await session.call_tool('web_search', {'query': '杭州今天天气'})
            print(res)


if __name__ == '__main__':
    asyncio.run(main())
```

如果我们发现在客户端有报错也不用慌，我们可以直接在日志中找到对应出错的请求点击`请求日志`查看报错来修复。

![image-20250406155803071](.assets/image-20250406155803071.png)

到这里，我们的 MCP 服务就被部署到了云端，我们就可以在任何地方直接来使用它了。

比如，在 `Cherry-Studio` 中，我们可以这样来设置：

![image-20250406160152782](.assets/image-20250406160152782.png)

在 `Cline` 中：

![image-20250406160709759](.assets/image-20250406160709759.png)

在 `Cursor` 中：

![image-20250406161055717](.assets/image-20250406161055717.png)

```json
{
  "mcpServers": {
    "web-search": {
      "url": "https://mcp-test-whhergsbso.cn-hangzhou.fcapp.run/sse"
    }
  }
}
```



至此，整个 MCP 入门教程就到这里啦，后续有其他的再进行更新。相关代码会放到 github 仓库中：https://github.com/liaokongVFX/MCP-Chinese-Getting-Started-Guide

