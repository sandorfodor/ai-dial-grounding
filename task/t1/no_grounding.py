import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }

# Initialize LLM client and token tracker
llm_client = AzureChatOpenAI(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="",
    model="gpt-4o"
)
token_tracker = TokenTracker()

def join_context(context: list[dict[str, Any]]) -> str:
    """Convert user data from JSON to readable string format."""
    result = []
    for user in context:
        user_info = "User:\n"
        for key, value in user.items():
            user_info += f"  {key}: {value}\n"
        result.append(user_info)
    return "\n".join(result)


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # Create messages array with system prompt and user message
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    # Generate response
    response = await llm_client.ainvoke(messages)
    
    # Get usage from response metadata
    total_tokens = response.response_metadata['token_usage']['total_tokens']
    
    # Add tokens to tracker
    token_tracker.add_tokens(total_tokens)
    
    # Print response and token usage
    print(f"Response: {response.content}")
    print(f"Tokens used: {total_tokens}")
    
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        # 1. Get all users
        user_client = UserClient()
        all_users = user_client.get_all_users()
        
        # 2. Split users into batches of 100
        batch_size = 100
        user_batches = [all_users[i:i + batch_size] for i in range(0, len(all_users), batch_size)]
        print(f"Split {len(all_users)} users into {len(user_batches)} batches")
        
        # 3. Prepare tasks for async execution
        tasks = []
        for batch in user_batches:
            context = join_context(batch)
            user_prompt = USER_PROMPT.format(context=context, query=user_question)
            tasks.append(generate_response(BATCH_SYSTEM_PROMPT, user_prompt))
        
        # 4. Run tasks asynchronously
        batch_results = await asyncio.gather(*tasks)
        
        # 5. Filter out NO_MATCHES_FOUND results
        filtered_results = [result for result in batch_results if "NO_MATCHES_FOUND" not in result]
        
        # 6. Generate final response or report no matches
        if filtered_results:
            # Combine filtered results
            combined_context = "\n\n".join(filtered_results)
            final_prompt = f"## SEARCH RESULTS:\n{combined_context}\n\n## USER QUESTION:\n{user_question}"
            
            print("\n--- Generating final answer ---")
            final_answer = await generate_response(FINAL_SYSTEM_PROMPT, final_prompt)
            print(f"\n=== FINAL ANSWER ===\n{final_answer}")
        else:
            print("No users found matching your criteria.")
        
        # 7. Print token usage summary
        summary = token_tracker.get_summary()
        print(f"\n=== TOKEN USAGE SUMMARY ===")
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Number of batches: {summary['batch_count']}")
        print(f"Tokens per batch: {summary['batch_tokens']}")


asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation