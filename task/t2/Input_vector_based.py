import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app

# System prompt explaining RAG context and how to use it
SYSTEM_PROMPT = """You are a RAG-powered assistant that helps users find information about users in the database.

## Structure of User message:
`RAG CONTEXT` - Retrieved user documents relevant to the query based on semantic similarity.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as the primary source when answering the `USER QUESTION`.
- Answer based ONLY on the information provided in the RAG context.
- If no relevant information exists in `RAG CONTEXT`, state that you cannot find matching users.
- Be specific and cite the user information when available.
- Format your response in a clear and organized manner.
"""

# User prompt template with RAG context and question
USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION:
{query}
"""


def format_user_document(user: dict[str, Any]) -> str:
    """Format user JSON data as a readable string for embeddings and context."""
    user_info = "User:\n"
    for key, value in user.items():
        user_info += f"  {key}: {value}\n"
    return user_info


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        # 1. Get all users
        user_client = UserClient()
        all_users = user_client.get_all_users()
        
        # 2. Prepare array of Documents
        documents = [Document(page_content=format_user_document(user)) for user in all_users]
        
        # 3. Create vectorstore with batching
        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        """Create FAISS vectorstore from documents in batches to avoid context window limits."""
        # 1. Split documents into batches
        doc_batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        print(f"Creating vectorstore from {len(documents)} documents in {len(doc_batches)} batches...")
        
        # 2. Create tasks to generate FAISS vector stores from each batch
        tasks = [
            FAISS.afrom_documents(batch, self.embeddings)
            for batch in doc_batches
        ]
        
        # 3. Gather all vectorstores
        vectorstores = await asyncio.gather(*tasks)
        
        # 4. Merge all vectorstores into one
        final_vectorstore = vectorstores[0]
        for vs in vectorstores[1:]:
            final_vectorstore.merge_from(vs)
        
        print("✅ Vectorstore created and merged successfully")
        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        """Retrieve relevant user documents using similarity search."""
        print(f"\n🔍 Searching for relevant users (top_k={k}, score_threshold={score})...")
        
        # 1. Make similarity search with relevance scores
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        
        # 2. Create context_parts array
        context_parts = []
        
        # 3. Iterate through retrieved docs
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"  - Score: {relevance_score:.4f} | {doc.page_content[:100]}...")
        
        # 4. Return joined context
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        """Augment user query with retrieved context."""
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        """Generate answer using LLM with augmented prompt."""
        # 1. Create messages array
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        
        # 2. Generate response
        response = self.llm_client.invoke(messages)
        
        # 3. Return response content
        return response.content


async def main():
    # 1. Create AzureOpenAIEmbeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        model="text-embedding-3-small-1",
        dimensions=384
    )
    
    # 2. Create AzureChatOpenAI
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        model="gpt-4o"
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            
            # 1. Retrieve context
            context = await rag.retrieve_context(user_question)
            
            if context:
                # 2. Make augmentation
                augmented_prompt = rag.augment_prompt(user_question, context)
                
                # 3. Generate answer and print it
                answer = rag.generate_answer(augmented_prompt)
                print(f"\n📝 Answer:\n{answer}\n")
            else:
                print("\n❌ No relevant users found matching your query.\n")


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce