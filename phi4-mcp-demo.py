from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion
from semantic_kernel.contents import ChatHistory, ChatMessageContent, ImageContent, TextContent
from semantic_kernel.connectors.mcp import MCPSsePlugin, MCPStdioPlugin
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments

import asyncio

async def main():
    kernel = Kernel()
    kernel.add_service(
        OllamaChatCompletion(
            host="http://localhost:11434/",
            ai_model_id="phi4-mini:3.8b-fp16"
        )
    )
    fetcher_mcp_plugin = MCPStdioPlugin(
        name="fetcher",
        description="Fetch web page from a URL",
        command="npx",
        args=["-y", "fetcher-mcp"]
    )
    await fetcher_mcp_plugin.connect()
    # playwright_mcp_plugin = MCPSsePlugin(
    #     name="playwright",
    #     description="Fetch web page from a URL",
    #     url="http://localhost:8931/sse"
    # )
    azure_mcp_plugin = MCPStdioPlugin(
        name="azure",
        description="MCP Server for azure resources and operations",
        command="npx",
        args=["-y", "@azure/mcp@latest", "server", "start"]
    )
    # await playwright_mcp_plugin.connect()
    kernel.add_plugin(fetcher_mcp_plugin, plugin_name="fetcher")
    # kernel.add_plugin(playwright_mcp_plugin, plugin_name="playwright")
    # kernel.add_plugin(azure_mcp_plugin, plugin_name="azure")
    settings = OllamaChatPromptExecutionSettings()
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="Assistant",
        instructions="Help the user with any task using your tools.",
        arguments=KernelArguments(settings=settings),
        plugins=[fetcher_mcp_plugin]
    )
    thread: ChatHistoryAgentThread = None

    is_complete = False
    while not is_complete:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            is_complete = True
            continue
        elif user_input.lower() == "clear":
            thread = None
            continue
        message = ChatMessageContent(
            role="user",
            items=[
                TextContent(text=user_input),
            ]
        )
        response = await agent.get_response(messages=message, thread=thread)
        thread = response.thread
        print("\n\n")
        print("--------------------------------------------------------------")
        print("Assistant: ", response.message.content)
        print("--------------------------------------------------------------")

    # message = ChatMessageContent(
    #     role="user",
    #     items=[
    #         TextContent(text="Summarize this page: https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/adding-mcp-plugins?pivots=programming-language-python"),
    #     ]
    # )
    # response = await agent.get_response(messages=message, thread=thread)

    # print("Assistant: ",response.message.content)
    await thread.delete() if thread else None
    await fetcher_mcp_plugin.close()
    # await playwright_mcp_plugin.close()

if __name__ == "__main__":
    asyncio.run(main())