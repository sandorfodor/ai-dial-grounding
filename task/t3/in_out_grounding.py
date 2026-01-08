import asyncio
from typing import Any, Optional
import json

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# HOBBIES SEARCHING WIZARD
# This implementation searches users by hobbies and provides their full info in JSON format
# Features:
# 1. Embeds only user `id` and `about_me` to reduce context window
# 2. Adaptive vector store updates - adds new users and removes deleted ones every request
# 3. Uses Named Entity Extraction (NEE) to extract hobby categories and matching user IDs
# 4. Output grounding - verifies user IDs and fetches complete user information
# 5. Returns JSON grouped by hobbies

SYSTEM_PROMPT = """You are a hobby extraction assistant. Your task is to analyze user profiles and extract hobbies mentioned in the search query.

## Instructions:
1. Review the user profiles provided in the RAG CONTEXT
2. Identify which users have hobbies matching or related to the search query
3. Group users by their specific hobby categories
4. Extract ONLY the user IDs for each hobby group
5. Return structured output with hobby names as keys and lists of user IDs as values

## Important:
- Extract specific hobby names from the user profiles (e.g., "rock climbing", "hiking", "camping")
- Group related hobbies under their specific categories
- Return ONLY user IDs, not full user data
- Be inclusive - if a user's hobby is related to the query, include them

## Output Format:
{format_instructions}
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUERY:
{query}

Extract user IDs grouped by their hobbies that match the query.
"""


class HobbyGroup(BaseModel):
    """Users grouped by a specific hobby."""
    hobby: str = Field(description="The specific hobby name (e.g., 'rock climbing', 'hiking')")
    user_ids: list[int] = Field(description="List of user IDs who have this hobby")


class HobbySearchResult(BaseModel):
    """Collection of hobby groups with matching users."""
    hobby_groups: list[HobbyGroup] = Field(
        default=[],
        description="List of hobby groups with their matching user IDs"
    )


def format_user_for_embedding(user: dict[str, Any]) -> str:
    """Format only user ID and about_me for embedding to reduce costs."""
    return f"User ID: {user['id']}\nAbout: {user.get('about_me', 'N/A')}"


class AdaptiveUserRAG:
    """RAG system with adaptive vector store updates and output grounding."""
    
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore: Optional[Chroma] = None
        self.user_client = UserClient()
        self.known_user_ids: set[int] = set()

    async def __aenter__(self):
        print("🔎 Initializing adaptive vector store...")
        # Load all users and create initial vectorstore
        all_users = self.user_client.get_all_users()
        
        # Prepare documents with only ID and about_me
        documents = [
            Document(
                page_content=format_user_for_embedding(user),
                metadata={"user_id": user["id"]},
                id=str(user["id"])  # Use user ID as document ID for easy updates
            )
            for user in all_users
        ]
        
        # Create Chroma vectorstore
        self.vectorstore = await Chroma.afrom_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="user_hobbies"
        )
        
        # Track known user IDs
        self.known_user_ids = {user["id"] for user in all_users}
        
        print(f"✅ Vectorstore initialized with {len(all_users)} users.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass

    async def update_vectorstore(self):
        """Update vectorstore by adding new users and removing deleted ones."""
        print("\n🔄 Updating vectorstore with latest user data...")
        
        # Get current users from API
        current_users = self.user_client.get_all_users()
        current_user_ids = {user["id"] for user in current_users}
        
        # Find new and deleted users
        new_user_ids = current_user_ids - self.known_user_ids
        deleted_user_ids = self.known_user_ids - current_user_ids
        
        # Remove deleted users
        if deleted_user_ids:
            print(f"  ➖ Removing {len(deleted_user_ids)} deleted users...")
            await self.vectorstore.adelete(ids=[str(uid) for uid in deleted_user_ids])
        
        # Add new users
        if new_user_ids:
            print(f"  ➕ Adding {len(new_user_ids)} new users...")
            new_users = [u for u in current_users if u["id"] in new_user_ids]
            new_documents = [
                Document(
                    page_content=format_user_for_embedding(user),
                    metadata={"user_id": user["id"]},
                    id=str(user["id"])
                )
                for user in new_users
            ]
            await self.vectorstore.aadd_documents(new_documents)
        
        # Update known user IDs
        self.known_user_ids = current_user_ids
        
        if not new_user_ids and not deleted_user_ids:
            print("  ✅ Vectorstore is up to date.")
        else:
            print(f"  ✅ Vectorstore updated: +{len(new_user_ids)} new, -{len(deleted_user_ids)} deleted")

    async def retrieve_relevant_users(self, query: str, k: int = 20) -> str:
        """Retrieve relevant user profiles using similarity search."""
        print(f"\n🔍 Searching for users with relevant hobbies (top_k={k})...")
        
        # Perform similarity search
        results = self.vectorstore.similarity_search(query, k=k)
        
        # Collect context
        context_parts = []
        for doc in results:
            context_parts.append(doc.page_content)
        
        print(f"  ✅ Found {len(results)} relevant user profiles")
        return "\n\n".join(context_parts)

    def extract_hobby_groups(self, query: str, context: str) -> HobbySearchResult:
        """Extract hobby groups with user IDs using structured output (NEE)."""
        print("\n🧠 Extracting hobby groups and user IDs...")
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=HobbySearchResult)
        
        # Create prompt with format instructions
        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(context=context, query=query))
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages=messages).partial(
            format_instructions=parser.get_format_instructions()
        )
        
        # Invoke LLM with structured output
        result: HobbySearchResult = (prompt | self.llm_client | parser).invoke({})
        
        print(f"  ✅ Extracted {len(result.hobby_groups)} hobby groups")
        for group in result.hobby_groups:
            print(f"    - {group.hobby}: {len(group.user_ids)} users")
        
        return result

    async def ground_output(self, hobby_result: HobbySearchResult) -> dict[str, list[dict[str, Any]]]:
        """
        Output grounding: Verify user IDs exist and fetch complete user information.
        Returns: Dictionary with hobby names as keys and full user data as values
        """
        print("\n🔐 Performing output grounding (verifying users and fetching full data)...")
        
        grounded_result = {}
        
        for group in hobby_result.hobby_groups:
            hobby = group.hobby
            user_ids = group.user_ids
            
            # Fetch full user data for each ID
            full_users = []
            for user_id in user_ids:
                try:
                    user_data = await self.user_client.get_user(user_id)
                    full_users.append(user_data)
                except Exception as e:
                    print(f"  ⚠️  User ID {user_id} not found (may have been deleted): {e}")
            
            if full_users:
                grounded_result[hobby] = full_users
                print(f"  ✅ {hobby}: Verified {len(full_users)}/{len(user_ids)} users")
        
        return grounded_result


async def main():
    # Initialize embeddings and LLM
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        model="text-embedding-3-small-1",
        dimensions=384
    )
    
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        model="gpt-4o"
    )

    async with AdaptiveUserRAG(embeddings, llm_client) as rag:
        print("\n" + "="*60)
        print("🎯 HOBBIES SEARCHING WIZARD")
        print("="*60)
        print("\nQuery samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find users interested in outdoor activities")
        print(" - Who likes technology and programming?")
        print("\nType 'quit' or 'exit' to stop.\n")
        
        while True:
            user_question = input("> ").strip()
            
            if user_question.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if not user_question:
                continue
            
            print("\n" + "-"*60)
            
            # Step 1: Update vectorstore with latest data
            await rag.update_vectorstore()
            
            # Step 2: Retrieve relevant user profiles (only ID and about_me)
            context = await rag.retrieve_relevant_users(user_question)
            
            if not context:
                print("\n❌ No relevant users found.\n")
                continue
            
            # Step 3: Extract hobby groups with user IDs (NEE with structured output)
            hobby_result = rag.extract_hobby_groups(user_question, context)
            
            if not hobby_result.hobby_groups:
                print("\n❌ No hobby groups extracted.\n")
                continue
            
            # Step 4: Output grounding - verify IDs and fetch full user data
            grounded_result = await rag.ground_output(hobby_result)
            
            # Step 5: Display results in JSON format
            if grounded_result:
                print("\n" + "="*60)
                print("📊 RESULTS (JSON Format)")
                print("="*60)
                print(json.dumps(grounded_result, indent=2))
                print("\n")
            else:
                print("\n❌ No valid users found after verification.\n")


asyncio.run(main())


# Benefits of Input-Output Grounding approach:
# ✅ Semantic search with vector embeddings (flexible queries)
# ✅ Real-time data - fetches live user information
# ✅ Adaptive updates - automatically syncs with user service changes
# ✅ Structured output - eliminates hallucinations with Pydantic models
# ✅ Output grounding - verifies all data is accurate and up-to-date
# ✅ Cost efficient - embeds only ID and about_me, not full profiles
# ✅ Named Entity Extraction - extracts only IDs, then fetches full data
# ✅ Prevents PII corruption - LLM only handles IDs, not sensitive data



