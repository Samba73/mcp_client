import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, HumanMessage
from dotenv import load_dotenv
import json

from regex import T

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
    
    },
    "expense": {
        "transport": "streamable_http",
        "url": "https://expense-tracking.fastmcp.app/mcp",
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

        # prompt = "what is the product of 13 and 47 using the math tool?"
        # prompt = "Treat as general query and response to what is capital of india?"
        # prompt = """Using the expense tool, list all the expenses available between 01-Nov-2025 and 22-Nov-2025.
        # Rs 1200 for transportation on 10-Nov-2025

        # using the expense tool.
        #             """
        prompt = "Using the expense tool, list all the expenses available between 01-Nov-2025 and 22-Nov-2025."
        # prompt = "Using both the math and expense tools, calculate the total expenses if I add Rs 1500 for rent and Rs 600 for utilities, and then multiply that total by 2."

        human_message = HumanMessage(content=prompt)
        response = await llm_with_tools.ainvoke([human_message])

        if not response.tool_calls:
            print(f"\nFinal response(No Tool Call): {response.content}")
            return
        tool_messages = []
        for tc in response.tool_calls:
            tool_to_use = response.tool_calls[0]["name"]
            tool_args = response.tool_calls[0]["args"]
            tool_id = response.tool_calls[0]["id"]
            # print(f"\nUsing tool: {tool_to_use}")

            tool_response = await named_tools[tool_to_use].ainvoke(tool_args)
                # print(f"\nTool response: {tool_response}")

            # tool_message = ToolMessage(content=tool_response, tool_call_id=tool_id)
            tool_messages.append(ToolMessage(content=json.dumps(tool_response), tool_call_id=tool_id))
        try:    
            final_response = await llm_with_tools.ainvoke([human_message, response, *tool_messages])
            print(f"\nFinal response: {final_response}")
            print(f"\nFinal response: {final_response.content}")
        except Exception as e:
            print(f"\nError during final response generation: {e}")    

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