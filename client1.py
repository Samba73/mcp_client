import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, HumanMessage
from dotenv import load_dotenv
import json

load_dotenv()


# The original command was trying to execute 'uv run fastmcp run ...' which is incorrect.
# We correct this to directly call the 'fastmcp' command with its 'run' arguments.
SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "uv",  # Correct command to start the fastmcp server
        "args": [
            "run",
            "fastmcp",
            "run",  # The subcommand for fastmcp to execute the application
            "C:\\Users\\Samba\\math\\main.py",  # The path to your fastmcp application
        ],
    }
}


async def main():
    print("Attempting to connect to MCP server via fastmcp...")
    
    # Initialize the client with the corrected configuration
    client = MultiServerMCPClient(SERVERS)
    
    try:
        # This call will attempt to start the process defined in SERVERS
        tools = await client.get_tools()

        print("\nConnection Successful!")
        # print("Available tools:", tools)

        named_tools = {}
        for tool in tools:
            named_tools[tool.name] = tool
            # print(f"\nTool Name: {tool.name}\nTool Details: {tool}")

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

        llm_with_tools = llm.bind_tools(tools)

        # prompt = "what is the product of 23 and 47 using the math tool?"
        prompt = "Treat as general query and response to what is capital of india?"

        human_message = HumanMessage(content=prompt)
        response = await llm_with_tools.ainvoke([human_message])

        if not response.tool_calls:
            print(f"\nFinal response: {response.content}")
            return

        tool_to_use = response.tool_calls[0]["name"]
        tool_args = response.tool_calls[0]["args"]
        tool_id = response.tool_calls[0]["id"]
        # print(f"\nUsing tool: {tool_to_use}")

        tool_response = await named_tools[tool_to_use].ainvoke(tool_args)
            # print(f"\nTool response: {tool_response}")

        tool_message = ToolMessage(content=tool_response, tool_call_id=tool_id)

        final_response = await llm_with_tools.ainvoke([human_message, response, tool_message])
        print(f"\nFinal response: {final_response.content}")

    except Exception as e:
        print("\n-------------------------------------------------")
        print("Connection Failed. Possible reasons:")
        print("1. 'fastmcp' is not installed or available in your current virtual environment.")
        print("2. The server file (main.py) has an error and is crashing on startup.")
        print("3. The path 'C:\\Users\\Samba\\math\\main.py' is incorrect.")
        print("-------------------------------------------------")
        # Re-raise the exception for the full traceback
        raise e


if __name__ == "__main__":
    asyncio.run(main())