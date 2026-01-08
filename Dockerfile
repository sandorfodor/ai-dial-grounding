FROM python:3.11
ADD . .
RUN pip install requests langchain-community langchain-openai langchain-chroma faiss-cpu python-dotenv aidial-client psycopg2-binary
CMD ["python", "./run.py"] 