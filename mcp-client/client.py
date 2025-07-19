import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

# import sys

load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')

        if not (is_python or is_js):
            raise ValueError("Server script must be a Python (.py) or JavaScript (.js) file.")
        
        command = 'python' if is_python else 'node'

        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,  # Use default environment variables
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        #list of available tools
        response = await self.session.list_tools()
        tools = response.tools
        print('connected to server with tools:', [tool.name for tool in tools])


    async def process_query(self, query: str)->str:
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            # "input_schema": tool.input_schema
        } for tool in response.tools]

        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        final_text = []

        assitant_message_content = []

        for content in response.content:
            if content.type == "text":
                final_text.append(content.text)
            elif content.type == "tool_use":
                tool_name = content.tool_name
                tool_args = content.input
                # print(f"Tool {tool_name} called with input: {tool_input}")
                
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"Calling Tool {tool_name} with args: {tool_args}")

                assitant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assitant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "result": result.content
                        }
                    ]
                })


                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(result.content[0].text)
        
        return '\n'.join(final_text)
    
    async def chat_loop(self):
        print("mcp-client started")
        print("Type 'exit' or 'quit' to stop the client.")
        while True:
            try:
                query = input("Enter your query: ").strip()
                if query.lower() in ['exit', 'quit']:
                    break
                response = await self.process_query(query)
                print("\n" + response)
            
            except Exception as e:
                print(f"An error occurred: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp-client.py <server_script_path>")
        return
    
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
        import sys
        asyncio.run(main())